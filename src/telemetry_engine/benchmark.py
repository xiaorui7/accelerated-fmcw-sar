"""Reproducible steady-state feature-engine and batch-size benchmarks."""
import argparse
import csv
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import threading
import time

import numba
import numpy as np
import psutil

from .features import get_backend, reference_features


def _cpu_name() -> str:
    if sys.platform == "win32":
        try:
            import winreg
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as key:
                return str(winreg.QueryValueEx(key, "ProcessorNameString")[0]).strip()
        except OSError:
            pass
    return platform.processor() or "unknown"


def _workload(cycles: int, channels: int, samples: int, seed: int = 2026) -> np.ndarray:
    rng = np.random.default_rng(seed)
    values = rng.normal(size=(cycles, channels, samples))
    values += np.arange(channels)[None, :, None] * 5
    return values


def _measure_memory(function, values, batch_size):
    process = psutil.Process()
    baseline = process.memory_info().rss
    samples = [baseline]
    stop = threading.Event()

    def monitor():
        while not stop.wait(.001):
            samples.append(process.memory_info().rss)

    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()
    try:
        for start in range(0, len(values), batch_size):
            function(values[start:start + batch_size])
        samples.append(process.memory_info().rss)
    finally:
        stop.set()
        thread.join()
    return max(samples) / 2**20, (max(samples) - baseline) / 2**20


def _worker(backend: str, threads: int, cycles: int, channels: int, samples: int,
            batch_size: int, repeats: int, oracle: str, streaming: bool = False) -> dict:
    if backend == "numba":
        numba.set_num_threads(min(threads, numba.config.NUMBA_NUM_THREADS))
    function = get_backend(backend)
    if streaming:
        warm_values = _workload(1, channels, samples)
        warm_started = time.perf_counter()
        function(warm_values)
        warmup = time.perf_counter() - warm_started
        check = _workload(min(batch_size, 4), channels, samples, seed=99)
        maximum_error = float(np.max(np.abs(function(check) - reference_features(check))))
        durations = []
        process = psutil.Process()
        baseline = process.memory_info().rss
        memory_samples = [baseline]
        stop = threading.Event()

        def monitor():
            while not stop.wait(.001):
                memory_samples.append(process.memory_info().rss)

        thread = threading.Thread(target=monitor, daemon=True)
        thread.start()
        try:
            for repeat in range(repeats):
                rng = np.random.default_rng(2026 + repeat)
                timed = 0.0
                for start in range(0, cycles, batch_size):
                    count = min(batch_size, cycles - start)
                    values = rng.normal(size=(count, channels, samples))
                    values += np.arange(channels)[None, :, None] * 5
                    started = time.perf_counter()
                    function(values)
                    timed += time.perf_counter() - started
                    memory_samples.append(process.memory_info().rss)
                durations.append(timed)
        finally:
            stop.set()
            thread.join()
        median = float(np.median(durations))
        return {
            "backend": backend, "threads": numba.get_num_threads() if backend == "numba" else 1,
            "cycles": cycles, "channels": channels, "samples_per_channel": samples,
            "batch_size": batch_size, "repeats": repeats, "median_runtime_s": median,
            "p95_runtime_s": float(np.percentile(durations, 95)),
            "throughput_cycles_s": cycles / median, "warmup_s": warmup,
            "max_abs_error": maximum_error, "sampled_peak_rss_mib": max(memory_samples) / 2**20,
            "sampled_rss_increase_mib": (max(memory_samples) - baseline) / 2**20,
        }
    values = _workload(cycles, channels, samples)
    warm_started = time.perf_counter()
    function(values[:1])
    warmup = time.perf_counter() - warm_started
    durations = []
    output = np.empty((cycles, channels, 8), dtype=np.float64)
    for _ in range(repeats):
        started = time.perf_counter()
        for start in range(0, cycles, batch_size):
            output[start:start + batch_size] = function(values[start:start + batch_size])
        durations.append(time.perf_counter() - started)
    if backend == "reference":
        np.save(oracle, output)
        maximum_error = 0.0
    else:
        expected = np.load(oracle, allow_pickle=False)
        maximum_error = float(np.max(np.abs(output - expected)))
    if maximum_error > 1e-10:
        raise ValueError(f"{backend} maximum absolute error {maximum_error} exceeds 1e-10")
    peak, increase = _measure_memory(function, values, batch_size)
    median = float(np.median(durations))
    return {
        "backend": backend, "threads": numba.get_num_threads() if backend == "numba" else 1,
        "cycles": cycles, "channels": channels, "samples_per_channel": samples,
        "batch_size": batch_size, "repeats": repeats, "median_runtime_s": median,
        "p95_runtime_s": float(np.percentile(durations, 95)),
        "throughput_cycles_s": cycles / median, "warmup_s": warmup,
        "max_abs_error": maximum_error, "sampled_peak_rss_mib": peak,
        "sampled_rss_increase_mib": increase,
    }


def _run_worker(output: Path, backend: str, threads: int, cycles: int, channels: int,
                samples: int, batch_size: int, repeats: int, oracle: str,
                streaming: bool = False) -> dict:
    command = [sys.executable, "-m", "telemetry_engine.benchmark", "--worker", backend,
               "--threads", str(threads), "--cycles", str(cycles), "--channels", str(channels),
               "--samples", str(samples), "--batch-size", str(batch_size),
               "--repeats", str(repeats), "--oracle", oracle]
    if streaming:
        command.append("--streaming")
    env = dict(os.environ, NUMBA_NUM_THREADS=str(max(threads, 1)), OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
    completed = subprocess.run(command, capture_output=True, text=True, env=env)
    if completed.returncode:
        raise RuntimeError(completed.stderr.strip() or completed.stdout.strip())
    return json.loads(completed.stdout)


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _report(rows: list[dict], batches: list[dict], environment: dict) -> str:
    text = "# Telemetry Engine Benchmark\n\n"
    text += "Steady-state timings cover aligned feature processing in bounded batches. Numba warm-up is excluded and reported separately.\n\n"
    text += "| Backend | Threads | Cycles | Channels × Samples | Batch | Median s | P95 s | Cycles/s | vs Reference | vs Numba 1T | Max abs error | Peak RSS MiB |\n"
    text += "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n"
    for row in rows:
        text += (f"| {row['backend']} | {row['threads']} | {row['cycles']} | {row['channels']} × {row['samples_per_channel']} | "
                 f"{row['batch_size']} | {row['median_runtime_s']:.6f} | {row['p95_runtime_s']:.6f} | "
                 f"{row['throughput_cycles_s']:.1f} | {row['speedup_vs_reference']:.2f}× | "
                 f"{row['speedup_vs_numba_1t']:.2f}× | {row['max_abs_error']:.2e} | {row['sampled_peak_rss_mib']:.1f} |\n")
    text += "\n## Batch-size study (Numba, 4 threads)\n\n"
    text += "| Batch size | Median s | Cycles/s | Peak RSS MiB | RSS increase MiB |\n|---:|---:|---:|---:|---:|\n"
    for row in batches:
        text += (f"| {row['batch_size']} | {row['median_runtime_s']:.6f} | {row['throughput_cycles_s']:.1f} | "
                 f"{row['sampled_peak_rss_mib']:.1f} | {row['sampled_rss_increase_mib']:.1f} |\n")
    text += ("\nThe batch study generates and releases one input batch at a time. Runtime includes feature calls only; "
             "sampled RSS includes batch generation and processing. Memory is sampled every 1 ms, so short peaks and allocator reuse can affect it.\n")
    return text


def run_benchmarks(output: str | Path, cycles: int = 1000, channels: int = 5,
                   samples: int = 600, batch_size: int = 128, repeats: int = 3,
                   batch_sizes: tuple[int, ...] = (32, 64, 128, 256)) -> dict:
    if min(cycles, channels, samples, batch_size, repeats, *batch_sizes) < 1:
        raise ValueError("benchmark sizes and repeats must be positive")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    with tempfile.TemporaryDirectory(prefix="telemetry-benchmark-") as temporary:
        oracle = str(Path(temporary) / "oracle.npy")
        cases = (("reference", 1), ("numpy", 1), ("numba", 1), ("numba", 2), ("numba", 4))
        for backend, threads in cases:
            rows.append(_run_worker(output, backend, threads, cycles, channels, samples,
                                    batch_size, repeats, oracle))
        reference_time = rows[0]["median_runtime_s"]
        numba_1t = next(row["median_runtime_s"] for row in rows if row["backend"] == "numba" and row["threads"] == 1)
        for row in rows:
            row["speedup_vs_reference"] = reference_time / row["median_runtime_s"]
            row["speedup_vs_numba_1t"] = numba_1t / row["median_runtime_s"]
        batch_rows = [_run_worker(output, "numba", 4, cycles, channels, samples, size,
                                  repeats, oracle, streaming=True) for size in batch_sizes]
    _write_csv(output / "benchmark.csv", rows)
    _write_csv(output / "batch_sizes.csv", batch_rows)
    environment = {
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "platform": platform.platform(),
        "processor": _cpu_name(), "logical_cpus": os.cpu_count(), "python": sys.version,
        "packages": {name: importlib.metadata.version(name) for name in ("numpy", "numba", "psutil")},
        "methodology": "Median and p95 steady-state aligned feature-engine wall time; Numba warm-up excluded; one process per case.",
    }
    (output / "benchmark_summary.json").write_text(json.dumps({"backends": rows, "batch_sizes": batch_rows, "environment": environment}, indent=2) + "\n", encoding="utf-8")
    (output / "benchmark_report.md").write_text(_report(rows, batch_rows, environment), encoding="utf-8")
    return {"backends": rows, "batch_sizes": batch_rows, "environment": environment}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", choices=("reference", "numpy", "numba"), required=True)
    parser.add_argument("--threads", type=int, required=True)
    parser.add_argument("--cycles", type=int, required=True)
    parser.add_argument("--channels", type=int, required=True)
    parser.add_argument("--samples", type=int, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--repeats", type=int, required=True)
    parser.add_argument("--oracle", required=True)
    parser.add_argument("--streaming", action="store_true")
    args = parser.parse_args()
    print(json.dumps(_worker(args.worker, args.threads, args.cycles, args.channels,
                             args.samples, args.batch_size, args.repeats, args.oracle, args.streaming)))
