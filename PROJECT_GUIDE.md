# Project Guide

Start with the call path in `telemetry_engine.cli.main()`. A `process` command loads an
`EngineConfig`, opens one dataset adapter, and passes its iterator to
`pipeline.process_to_directory()`. The pipeline validates IDs and channel data, aligns
one batch, selects a function from `features.BACKENDS`, applies `rules.apply_rules()`,
and writes results before requesting the next batch.

The central engineering choice is the reference implementation. Read
`reference_features()` first, then compare `numpy_features()` and `_numba_kernel()`.
They calculate the same eight values but trade readability for lower interpreter
overhead and CPU parallelism. The tests and benchmark convert that shared contract into
evidence rather than assuming faster code is correct.

Use these exercises before an interview:

1. Trace one synthetic cycle from its NPZ file to one CSV row.
2. Explain why the source adapter and internal model are separate.
3. Change target rate from 10 Hz to 5 Hz and predict aligned array shape.
4. Break one feature formula and identify the tests and benchmark guard that fail.
5. Compare batches 32 and 256 and explain both throughput and RSS results.
6. Run Numba with one and four threads and explain why each output row has one owner.
7. Add a new threshold rule without touching any numerical backend.
8. State the limitations in the README without overstating the UCI demonstration.
