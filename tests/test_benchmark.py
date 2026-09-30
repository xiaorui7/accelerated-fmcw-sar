import csv
import json
from sar.benchmark import run_benchmarks


def test_benchmark_outputs(tmp_path):
    output = tmp_path / "benchmark.csv"
    rows = run_benchmarks(output, [4], [2], repeats=1, threads=1)
    assert len(rows) == 3
    assert all(r["runtime_s"] > 0 and r["nrmse"] <= 1e-10 for r in rows)
    assert rows[0]["speedup"] == 1
    with output.open() as stream:
        assert len(list(csv.DictReader(stream))) == 3
    assert output.with_suffix(".png").is_file()
    assert output.with_suffix(".md").is_file()
    assert json.loads(output.with_suffix(".environment.json").read_text())["repeats"] == 1
