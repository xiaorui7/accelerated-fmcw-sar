# Project Guide

This guide is for learning and interview preparation. Read it next to the code, run
the examples, and change one parameter at a time. You should be able to explain each
module below without relying on the wording in this document.

## The project in one minute

The program turns either a configured point-target scene or stored radar sweeps into
a two-dimensional complex image. It separates the workflow into simulation, FFT range
processing, geometry, Back-Projection, metrics, and output. The same BP interface has
a readable scalar implementation, a chunked NumPy implementation, and a compiled
Numba implementation. Tests check the optimized results against the reference and
also check parts of the physical model independently.

The main engineering problem is not inventing a new radar algorithm. It is building a
small numerical pipeline whose optimized implementation remains measurable,
reproducible, and correct within a documented tolerance.

## Follow one reconstruction

Run:

```bash
python -m sar.cli reconstruct --config configs/single_target.yaml --backend numba
```

The call path is:

1. `cli.main()` loads and validates YAML into a `Config`.
2. `pipeline.reconstruct()` asks `scene.py` and `geometry.py` for target and aperture
   coordinates.
3. `fmcw.simulate()` generates a complex dechirped sweep at every aperture position.
4. `range_compression.range_compress()` windows each sweep and applies an FFT.
5. `geometry.image_grid()` creates the candidate image pixels.
6. The selected function in `backprojection.BACKENDS` interpolates each range profile,
   compensates its phase, and accumulates aperture contributions.
7. The CLI saves the complex image as NPZ and asks `visualization.py` for a PNG.

When `--input` is supplied, step 3 is replaced by `io.load_data()`. The archive's radar
metadata overrides the YAML radar section so the data is processed with the parameters
that produced it.

## Configuration and scenes

Important files: `src/sar/config.py`, `src/sar/scene.py`, and `configs/*.yaml`.

`RadarConfig`, `GridConfig`, and `Config` are frozen dataclasses. They group related
values, calculate chirp slope, and reject values that would make the numerical code
ambiguous. `load_config()` uses `yaml.safe_load`; unknown YAML keys fail instead of
being silently ignored.

`make_scene()` returns rows of `(x, y, amplitude)`. The built-in scenes are a single
target and an illustrative 5x5 target grid. A custom configuration supplies its own
rows.

Example input:

```yaml
scene: custom
pulses: 32
targets:
  - [0.0, 10.0, 1.0]
```

Output: a NumPy array with shape `(number_of_targets, 3)`.

Failure cases: malformed YAML, unknown fields, invalid grid bounds, too few pulses,
nonpositive radar parameters, negative jitter, and missing custom targets.

Interview question: **Why use dataclasses instead of dictionaries everywhere?**

Answer: The fields and defaults form an explicit contract, validation happens once at
the boundary, public functions get useful type hints, and a frozen object cannot be
silently changed in the middle of a run.

## Geometry

Important file: `src/sar/geometry.py`.

`trajectory()` creates a straight, evenly spaced aperture. `image_grid()` converts
configured axis bounds into both plotting axes and flattened `(x, y)` pixels.
`ranges()` computes pairwise Euclidean distances.

Concrete example: the distance from `(0, 0)` to `(3, 4)` is exactly 5 m. The geometry
test uses cases like this so a vectorized implementation is checked against values
known without running the radar pipeline.

Failure handling is mostly at the configuration and caller boundaries. These functions
remain small because the equations are their interface; wrapping them in geometry
classes would not add state or behavior.

Interview question: **Why flatten image pixels?**

Answer: Each BP backend can accept one `(pixels, 2)` array regardless of image shape.
The pipeline reshapes the one-dimensional result only after reconstruction, which
keeps the kernels simple.

## FMCW simulation and range processing

Important files: `src/sar/fmcw.py` and `src/sar/range_compression.py`.

For a target range `R`, the round-trip delay is `tau = 2R/c`. `simulate()` evaluates
the analytic dechirped phase for every aperture, target, and fast-time sample. It
rejects a scene whose beat frequency reaches the positive Nyquist boundary.

`range_compress()` multiplies sweeps by a Hann window, zero-pads them, runs a SciPy
FFT, and maps frequency bins to range. It also removes the Hann window's center-time
phase ramp before interpolation. `phase_coefficients()` returns the linear and
quadratic range-phase coefficients that BP later cancels.

Input: a finite complex array shaped `(pulses, samples)` plus `RadarConfig`.

Output: contiguous complex range profiles and a uniformly spaced range axis.

Failure cases: wrong sweep shape, nonfinite samples, and aliased target ranges.

Interview question: **Does zero-padding improve physical range resolution?**

Answer: No. Bandwidth and windowing determine physical resolution. Zero-padding
samples the FFT response more densely, which helps interpolation but adds no measured
information.

## Back-Projection implementations

Important file: `src/sar/backprojection.py`.

Every backend receives range profiles, their range bins, aperture positions, flattened
pixels, and two phase coefficients. For each pixel and aperture it:

1. computes distance from aperture to pixel;
2. maps the distance to a fractional range-bin index;
3. linearly interpolates the complex profile;
4. cancels the predicted phase; and
5. averages the coherent sum over apertures.

`bp_reference()` uses explicit Python loops. It is intentionally easy to trace.
`bp_vectorized()` loops over apertures but processes up to 8,192 pixels with NumPy at
a time. `bp_numba()` compiles the scalar structure and uses `prange` over pixels.

Out-of-range samples contribute zero. Input checks reject inconsistent dimensions,
nonfinite values, nonuniform range bins, and invalid chunk sizes.

Interview question: **Why is the Numba loop free of data races?**

Answer: Each parallel iteration owns one output index. It accumulates that pixel's
pulse contributions in a local scalar and writes the result once. No threads update
the same pixel.

Interview question: **Why keep the vectorized implementation when Numba is faster?**

Answer: It demonstrates a portable optimization that does not depend on JIT-compiled
loops and exposes the memory trade-off. It also provides a second optimized
implementation for cross-checking behavior.

Interview question: **What does the 517.9x result mean?**

Answer: On the saved machine and workload, warmed BP-only median time fell from 47.691
seconds in the deliberately simple scalar Python reference to 0.092 seconds with four
Numba threads. It is not end-to-end latency or a comparison with optimized native code.

## Motion experiment and metrics

Important files: `src/sar/motion.py`, `src/sar/metrics.py`, and `src/sar/pipeline.py`.

`perturb()` adds a seeded Gaussian position error and a fixed offset. The pipeline
simulates sweeps at the actual perturbed positions, reconstructs them using the nominal
positions, and reconstructs the same sweeps using the known actual positions.

Outputs include three images, normalized error, peak location, peak magnitude, and
3-dB widths. The metrics module returns `None` when a lobe crossing falls outside the
image instead of inventing a width.

Failure cases: negative jitter, invalid offsets, mismatched correction shapes, image
shape mismatch, and a zero reference image.

Interview question: **Is this autofocus?**

Answer: No. The code is given the exact simulated trajectory error. It demonstrates
the sensitivity of coherent processing to geometry and verifies correction plumbing;
it does not estimate an unknown path.

## Storage and CLI

Important files: `src/sar/io.py` and `src/sar/cli.py`.

The NPZ archive contains `schema`, `data`, `positions`, and JSON radar metadata.
`allow_pickle=False` prevents object deserialization. `load_data()` verifies the schema,
required fields, shapes, and finite values.

The CLI turns library functions into reproducible workflows. Expected user errors are
reported through `argparse` with a nonzero exit rather than an internal traceback.
Unexpected programming errors are allowed to surface during development.

Interview question: **Why NPZ instead of a database?**

Answer: The project stores a few homogeneous arrays for one local experiment. NPZ is
portable, compressed, directly supported by NumPy, and easy to version. There is no
query, transaction, concurrent-user, or remote-access requirement that would justify
a database.

## Benchmarking

Important files: `src/sar/benchmark.py` and `benchmarks/profile_reference.py`.

The initial cProfile run showed most reference time inside the scalar BP loop. The
benchmark runner then compares backends at identical grid and pulse counts. It uses a
fresh subprocess for each case, performs a small warm-up, times repeated full calls,
and compares optimized output with a reference array outside the timed region.

It writes CSV rows, a Markdown table, a plot, and environment metadata. Memory is
sampled in a separate warm run approximately every millisecond. That measurement is an
estimate of process RSS, not an exact allocation peak.

Failure cases: invalid benchmark arguments, numerical error above `1e-10`, or a worker
process failure. Worker errors include the backend and workload that failed.

Interview question: **Why exclude JIT warm-up?**

Answer: The benchmark measures steady-state kernel execution and stores warm-up time
separately. A user concerned with one-shot latency should add compilation time; the
README therefore labels the saved numbers as warmed BP-only timing.

## Visualization

Important file: `src/sar/visualization.py`.

Plotting is kept outside the numerical kernels so tests can compare complex arrays
without rendering figures. `magnitude_db()` applies a configurable floor, while
`plot_images()` uses a shared peak across comparison panels so motion-induced peak
loss remains visible.

Interview question: **Why save the complex image as well as a PNG?**

Answer: The PNG is for inspection. The complex array retains phase and precision for
metrics, comparisons, and later processing.

## Tests and CI

Important files: `tests/` and `.github/workflows/tests.yml`.

The most important protections are:

- analytic geometry catches coordinate and broadcasting errors;
- range/phase tests catch sign and FFT-convention mistakes;
- direct matched filtering gives an independent check on interpolated BP;
- backend agreement catches optimization regressions;
- motion tests verify that error degrades focus and known correction improves it;
- NPZ tests reject damaged, incomplete, and nonfinite input;
- CLI tests cover integration among configuration, storage, processing, and figures;
- the benchmark smoke test verifies generated machine-readable and visual artifacts.

The workflow runs these tests on Linux and Windows with Python 3.11 and 3.13. A workflow
file is configuration, not evidence of a successful hosted run; check the repository's
Actions page after publishing.

## Scope you should state clearly

- The current data is synthetic.
- The geometry is planar, monostatic, and stop-and-go.
- There is no propagation loss, antenna response, receiver noise, calibration,
  multipath, GPU path, FFBP, or estimated autofocus.
- The original undergraduate MATLAB files were not available for comparison.
- AI assisted implementation and review; ownership means understanding, testing, and
  being able to change the resulting code, not pretending it was produced differently.

## Exercises before an interview

1. Draw the call path from the CLI to `bp_numba()` without looking at this guide.
2. Change `chunk_size` and explain which memory allocation it bounds.
3. Run the benchmark with one and four Numba threads and explain the comparison.
4. Break the phase sign in one backend and identify which tests fail.
5. Add a second custom target in YAML and predict the shape of every major array.
6. Explain why backend agreement alone cannot prove the physical equations are right.
