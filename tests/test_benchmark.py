import csv
import json

from telemetry_engine.benchmark import run_benchmarks


def test_benchmark_artifacts(tmp_path):
    result = run_benchmarks(tmp_path, cycles=4, channels=2, samples=8, batch_size=2,
                            repeats=1, batch_sizes=(1, 2))
    assert len(result["backends"]) == 5
    assert all(row["max_abs_error"] <= 1e-10 for row in result["backends"])
    assert len(list(csv.DictReader((tmp_path / "benchmark.csv").open()))) == 5
    assert len(json.loads((tmp_path / "benchmark_summary.json").read_text())["batch_sizes"]) == 2
    assert (tmp_path / "benchmark_report.md").is_file()
