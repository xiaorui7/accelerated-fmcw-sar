# Final Audit

## A. System summary

The repository is a runnable Python package and CLI for ingesting heterogeneous
industrial-equipment cycles, validating them, aligning multi-rate channels, calculating
eight features through interchangeable backends, evaluating threshold rules, and writing
machine-readable results in bounded batches.

## B. Architecture

```text
UCI/synthetic files -> adapter -> TelemetryCycle -> validation -> alignment
-> Reference/NumPy/Numba features -> diagnostic rules -> CSV/JSON/metadata
```

## C. Code changes

- `model.py`: internal `TelemetryChannel`, `TelemetryCycle`, and `AlignedBatch` contracts.
- `datasets/hydraulic.py`: streaming UCI parser, sensor metadata, sample-count checks.
- `datasets/synthetic.py`: deterministic multi-rate fixture generator and reader.
- `validation.py`, `alignment.py`: explicit failure behavior and resampling.
- `features.py`: common Reference, NumPy, and parallel Numba feature interface.
- `pipeline.py`: duplicate detection, batching, incremental artifact writing, metadata.
- `rules.py`: configuration-driven min/max diagnostics.
- `benchmark.py`: isolated timing, p95, throughput, correctness, and RSS collection.
- `cli.py`: five developer commands with meaningful exit code 2 for invalid input.
- Old radar source, configs, figures, benchmark claims, docs, and tests were removed.

## D. Real data support

Place selected UCI sensor files in one directory. `HydraulicDataset.iter_cycles()` opens
them together, consumes one row from each, checks that all files have the same cycle
count, verifies 60 seconds of samples at each documented rate, parses `profile.txt` when
present, maps names/units, and yields the internal model without loading the full archive.

## E. Test results

```text
python -m pytest --cov=telemetry_engine --cov-report=term-missing -q
40 passed in 14.79s
TOTAL 773 statements, 186 missed, 76% coverage
```

Runtime varies; counts and coverage are the reproducible result.

## F. Performance results

| Backend | Threads | Median s | Cycles/s | Speedup vs reference |
|---|---:|---:|---:|---:|
| Reference | 1 | 1.254950 | 796.8 | 1.00× |
| NumPy | 1 | 0.178336 | 5,607.4 | 7.04× |
| Numba | 1 | 0.195116 | 5,125.1 | 6.43× |
| Numba | 2 | 0.074541 | 13,415.5 | 16.84× |
| Numba | 4 | 0.045894 | 21,789.2 | 27.34× |

Workload: 1,000 cycles, 5 channels, 600 samples/channel, batch 128, three repetitions.
Machine: Intel Core Ultra 9 285H, Windows 11, Python 3.13.12, NumPy 2.4.4, Numba 0.68.0.

## G. Memory/batch results

| Batch | Cycles/s | Peak RSS MiB | RSS increase MiB |
|---:|---:|---:|---:|
| 32 | 20,428.3 | 108.8 | 1.7 |
| 64 | 27,258.9 | 110.4 | 3.9 |
| 128 | 18,219.0 | 113.1 | 6.4 |
| 256 | 20,417.2 | 118.9 | 12.1 |

Batch 32 reduced sampled peak RSS by 10.4 MiB versus 256; batch 64 was fastest in this
run. RSS is sampled rather than an exact allocation trace.

## H. Correctness results

NumPy maximum absolute error versus Reference was `4.62e-14`; all Numba thread cases
were `2.22e-16`. Both are below the benchmark threshold of `1e-10`.

## I. Claim audit

| Resume claim | Exact source file/function | Supporting test or benchmark |
|---|---|---|
| Validates heterogeneous telemetry | `validation.py: validate_channel`, `validate_cycle` | `test_model_validation.py` failure-path tests |
| Aligns multi-rate channels | `alignment.py: align_cycles`, `_resample` | `test_linear_alignment`, `test_nearest_alignment` |
| Supports real UCI hydraulic cycles | `datasets/hydraulic.py: HydraulicDataset.iter_cycles` | `test_hydraulic_adapter_tiny_fixture`, malformed-row test |
| Provides three interchangeable backends | `features.py: BACKENDS`, three feature functions | `test_backend_agreement`; `benchmark.csv` |
| Parallelizes safely across CPU threads | `features.py: _numba_kernel` | 1T/2T/4T rows in `benchmark.csv` |
| Processes in bounded batches | `pipeline.py: batched`, `process_to_directory` | `test_batched_sizes`; `batch_sizes.csv` |
| Produces structured outputs and metadata | `pipeline.py: process_to_directory` | `test_structured_outputs`, CLI integration test |
| Applies configurable diagnostics | `rules.py: apply_rules` | `test_rule_flags_high_value`, invalid-key test |
| Has 40 passing tests and 76% coverage | `tests/`, pytest configuration | saved command/result in this audit |

## J. Limitations

Do not claim production or factory deployment, users, saved money/time, downtime
reduction, diagnostic accuracy, predictive maintenance, distributed scale, or exact
memory allocation measurement. The performance study measures warmed aligned feature
processing; it is not end-to-end file-to-output latency.
