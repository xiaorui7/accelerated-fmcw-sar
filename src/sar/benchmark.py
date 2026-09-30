"""Isolated-process BP measurements; compilation and memory sampling are separate."""
import argparse
import csv
import importlib.metadata
import json
import logging
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import threading
import time

import numpy as np
import numba
import psutil

from .backprojection import BACKENDS
from .config import Config, GridConfig
from .fmcw import simulate
from .geometry import trajectory, image_grid
from .io import write_json
from .metrics import normalized_error
from .range_compression import range_compress, phase_coefficients
from .scene import make_scene

LOGGER = logging.getLogger(__name__)


def _worker(backend: str, size: int, pulses: int, repeats: int, threads: int, oracle_path: str) -> dict:
    numba.set_num_threads(min(threads, numba.config.NUMBA_NUM_THREADS))
    config = Config(grid=GridConfig(size=size), pulses=pulses, scene="single")
    positions = trajectory(config)
    profiles, bins = range_compress(simulate(make_scene(config), positions, config.radar), config.radar)
    args = (profiles, bins, positions, image_grid(config.grid)[2], phase_coefficients(config.radar))
    function = BACKENDS[backend]
    # Warm-up on identical dtypes but a tiny pixel set; record separately.
    started = time.perf_counter()
    function(*args[:3], args[3][:4], args[4])
    warmup = time.perf_counter() - started
    durations = []
    for _ in range(repeats):
        started = time.perf_counter()
        result = function(*args)
        durations.append(time.perf_counter() - started)
    if backend == "reference":
        np.save(oracle_path, result)
        error = 0.0
    else:
        error = normalized_error(result, np.load(oracle_path, allow_pickle=False))
    if error > 1e-10:
        raise ValueError(f"{backend} failed numerical tolerance: {error}")
    del result
    process = psutil.Process()
    baseline = process.memory_info().rss
    samples = [baseline]
    stop = threading.Event()

    def sample() -> None:
        while not stop.wait(.001):
            samples.append(process.memory_info().rss)

    monitor = threading.Thread(target=sample, daemon=True)
    monitor.start()
    try:
        memory_result = function(*args)  # Hold output through the final RSS sample.
        samples.append(process.memory_info().rss)
    finally:
        stop.set()
        monitor.join()
    return {"implementation": backend, "grid_size": size, "pulses": pulses,
            "runtime_s": float(np.median(durations)), "min_s": min(durations), "max_s": max(durations),
            "repeats": repeats, "warmup_s": warmup, "nrmse": error,
            "sampled_peak_rss_mib": max(samples) / 2**20,
            "sampled_rss_increase_mib": (max(samples) - baseline) / 2**20,
            "threads": numba.get_num_threads() if backend == "numba" else 1}


def render_report(rows: list[dict], output: Path) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), layout="constrained")
    for pulses in sorted({int(row["pulses"]) for row in rows}):
        for backend in BACKENDS:
            group = [r for r in rows if r["pulses"] == pulses and r["implementation"] == backend]
            axes[0].plot([r["grid_size"] for r in group], [r["runtime_s"] for r in group], "o-",
                         label=f"{backend}, {pulses} pulses")
            if backend != "reference":
                axes[1].plot([r["grid_size"] for r in group], [r["speedup"] for r in group], "o-",
                             label=f"{backend}, {pulses} pulses")
    axes[0].set(xlabel="Grid side (pixels)", ylabel="Median BP time (s)", yscale="log")
    axes[1].set(xlabel="Grid side (pixels)", ylabel="Speedup vs scalar reference", yscale="log")
    for ax in axes:
        ax.grid(alpha=.2)
        ax.legend(fontsize=7)
    fig.savefig(output.with_suffix(".png"), dpi=160)
    plt.close(fig)
    table = "| Grid | Pulses | Backend | Median seconds | Speedup | Complex NRMSE | Sampled RSS MiB |\n"
    table += "|---|---:|---|---:|---:|---:|---:|\n"
    for r in rows:
        table += (f"| {r['grid_size']}² | {r['pulses']} | {r['implementation']} | {r['runtime_s']:.6f} | "
                  f"{r['speedup']:.2f}× | {r['nrmse']:.2e} | {r['sampled_peak_rss_mib']:.1f} |\n")
    output.with_suffix(".md").write_text(table, encoding="utf-8")
    return table


def run_benchmarks(output: str | Path, sizes: list[int], pulses: list[int],
                   repeats: int = 3, threads: int = 4) -> list[dict]:
    if repeats < 1 or threads < 1 or not sizes or not pulses or min(sizes + pulses) < 2:
        raise ValueError("sizes/pulses >= 2, repeats/threads >= 1 are required")
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    with tempfile.TemporaryDirectory(prefix="sar-benchmark-") as temporary:
        oracle_path = str(Path(temporary) / "oracle.npy")
        for size in sizes:
            for count in pulses:
                baseline = None
                for backend in BACKENDS:
                    LOGGER.info("Benchmark %s: %sx%s, %s pulses", backend, size, size, count)
                    command = [sys.executable, "-m", "sar.benchmark", "--worker", backend,
                               "--size", str(size), "--pulses", str(count), "--repeats", str(repeats),
                               "--threads", str(threads), "--oracle", oracle_path]
                    env = dict(os.environ, NUMBA_NUM_THREADS=str(threads), OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
                    completed = subprocess.run(command, capture_output=True, text=True, env=env)
                    if completed.returncode != 0:
                        details = completed.stderr.strip() or completed.stdout.strip() or "no subprocess output"
                        raise RuntimeError(
                            f"{backend} benchmark failed for {size}x{size}, {count} pulses: {details}"
                        )
                    row = json.loads(completed.stdout)
                    if baseline is None:
                        baseline = row["runtime_s"]
                    row["speedup"] = baseline / row["runtime_s"]
                    rows.append(row)
                    with output.open("w", newline="", encoding="utf-8") as stream:
                        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                        writer.writeheader()
                        writer.writerows(rows)
    write_json(output.with_suffix(".environment.json"), {
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "platform": platform.platform(),
        "processor": platform.processor(), "logical_cpus": os.cpu_count(), "python": sys.version,
        "packages": {name: importlib.metadata.version(name) for name in ("numpy", "scipy", "numba", "psutil")},
        "sizes": sizes, "pulses": pulses, "repeats": repeats, "numba_threads": threads,
        "timing": "Median BP-only wall time after tiny-input warm-up; fresh process per backend/case.",
        "memory": "Separate warm run, RSS sampled every 1 ms. Includes interpreter/libraries/input; allocator reuse and missed short peaks possible."})
    render_report(rows, output)
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", choices=BACKENDS, required=True)
    parser.add_argument("--size", type=int, required=True)
    parser.add_argument("--pulses", type=int, required=True)
    parser.add_argument("--repeats", type=int, required=True)
    parser.add_argument("--threads", type=int, required=True)
    parser.add_argument("--oracle", required=True)
    a = parser.parse_args()
    print(json.dumps(_worker(a.worker, a.size, a.pulses, a.repeats, a.threads, a.oracle)))
