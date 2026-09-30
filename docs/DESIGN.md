# Numerical contract

All coordinates are metres, frequencies hertz, times seconds. The acquisition is
planar, monostatic and stop-and-go: the platform does not move during a sweep.
The image is a planar slice, with `x` across the aperture and positive `y` down range.
`carrier_hz` means the chirp **start frequency**, not its centre frequency.

## Echo simulation: `fmcw.py`

For aperture position a_m and target p_q, let

```text
R_mq = ||p_q - a_m||
tau_mq = 2 R_mq / c
K = B / T
fs = N / T
t_n = n / fs, n = 0,...,N-1
s_m[n] = sum_q A_q exp(j 2π [fc tau_mq + K tau_mq t_n - K tau_mq² / 2])
```

This is transmit times conjugated delayed receive, an analytic dechirped signal with
positive beat frequency. It retains residual video phase. The phase formula assumes
usable, continuously defined chirps over the relevant delay; chirp-edge gating is not
modelled. Amplitudes are constant scattering coefficients; there is no range loss,
antenna pattern, receiver noise, multipath or hardware calibration.

## Range compression: `range_compression.py`

Multiply by a symmetric Hann window, zero-pad to L = N × zero_pad, and take a forward
FFT divided by the window sum. Keep positive-frequency bins below Nyquist. The range
axis is R_k = c f_k/(2K), with f_k = k fs/L. Scene beat frequencies at or beyond fs/2
are rejected. Nominal range resolution is c/(2B), whereas FFT range-bin spacing is
c/(2B × zero_pad). Zero padding improves interpolation sampling; it does not add
physical resolution. The Hann window broadens the main lobe.

Before interpolation multiply each FFT bin by exp(+j 2π f_k t_c), where
t_c = (N-1)/(2fs). The symmetric window's spectrum has a linear phase ramp. Removing
that ramp makes complex linear interpolation less sensitive to rapid bin-phase
rotation. At a target's beat frequency, the resulting profile has phase

```text
phi(R) = 4π(fc + K t_c) R/c - 4π K R²/c² = k1 R + k2 R².
```

Both the window-centre phase and residual video phase must therefore be compensated
in BP. Omitting either while changing the FFT convention is a correctness error.

## Backprojection: `backprojection.py`

```text
u = (||p - a_m|| - R_0) / ΔR
l = floor(u), alpha = u - l
q_m(R) = (1-alpha) profile[m,l] + alpha profile[m,l+1]
I(p) = (1/M) sum_m q_m(||p-a_m||) exp(-j phi(||p-a_m||))
```

Interpolation returns zero outside the stored range support, and includes both end
bins. Mean accumulation normalizes an on-bin unit target to unit magnitude. Physical
range and phase calculations use float64; profiles/images use complex128.

The scalar reference independently expresses pixel and aperture loops using Python
and `math`. Vectorization operates on up to 8192 pixels per chunk and loops over
apertures, avoiding a full M-by-P geometry allocation. Numba uses `prange` over
pixels; each thread owns its pixel and accumulates pulses sequentially. No fastmath,
approximate square roots, reduced precision or shared pixel writes are used.
All backends have O(MP) work; optimizations reduce constant costs, not asymptotic work.
Input profiles use O(ML), pixel coordinates/output O(P); vectorized scratch O(chunk).
The Numba kernel uses O(P) output and per-thread scalar scratch.

Correctness contract: complex relative L2 error <= 1e-10 versus reference. Small
tests additionally require `allclose(rtol=1e-10, atol=1e-11)`. This is an implementation
agreement tolerance, not a physical accuracy claim. Analytic geometry, on-bin FFT phase,
off-bin matched-filter comparison, interpolation boundaries and single-target
localization test assumptions independently of backend agreement.

## Motion experiment: `motion.py`, `pipeline.py`

Actual positions = nominal positions + systematic offset + seeded independent Gaussian
jitter per coordinate. Ideal acquisition uses nominal positions for both simulation
and reconstruction. Motion-corrupted acquisition uses actual positions for simulation
but nominal positions for BP. Known-error correction supplies nominal + known error
to BP on those **same corrupted-acquisition sweeps**.

The actual-geometry oracle and corrected path intentionally coincide. Zero corrected
error to that oracle verifies plumbing; it is not evidence of an estimated motion
algorithm. Even exact correction differs from ideal acquisition because it samples
a different aperture. The figure and JSON report both comparisons.

## Metrics: `metrics.py`

| Metric | Definition and interpretation |
|---|---|
| Complex normalized reconstruction error | ||I - I_ref||₂ / ||I_ref||₂. No phase alignment, amplitude scaling or registration. Rejects a zero reference. |
| Localization error | Euclidean distance from the global magnitude peak pixel to a known isolated target. Limited by grid spacing. |
| Peak magnitude | Maximum abs(I), with mean pulse and window normalization. Off-bin interpolation and incoherence can lower it. |
| 3-dB width | Contiguous region around the peak above peak/sqrt(2), measured on x/y cuts through the image peak. Linear interpolation finds crossings; null means a crossing is outside the grid or there is no signal. |
| Runtime | Median of repeated BP-only `perf_counter` measurements; includes input validation and output allocation, excludes simulation/FFT/plotting and tiny-input warm-up. Min/max and warm-up seconds are also stored. |
| Speedup | Reference median / backend median for the same grid and pulse count. |
| Sampled peak memory | Maximum process RSS observed every 1 ms during a separate warm reconstruction; absolute RSS and increase over its pre-call baseline. Includes libraries/input and allocator retention; brief peaks can be missed. |

Global-peak localization and widths are restricted to single-target reports. They
would conflate targets and sidelobes in the 25-target scene. PSLR is omitted because
robust main-lobe exclusion and overlapping-target handling have not been implemented.

## Benchmark method

Each backend/case runs in a fresh subprocess on the same deterministic single-target
signal. Reference runs first and writes the complex oracle for the other backends.
Numba initialization/cache loading is timed separately on a four-pixel input before
three full timed calls. Four Numba threads are requested; NumPy uses one BLAS/OpenMP
thread. Output comparison occurs outside the timed region. Timing and memory sampling
are separate to avoid sampler overhead in latency measurements. Tiny-input warm-up
does not pre-touch the full image: min/max expose variation, and results are host-specific.
No claim is made that all host background activity is controlled.

The pre-optimization cProfile record is `benchmarks/profile_reference.txt`: 0.451 s
self-time in BP out of 0.557 s cumulative on 64² pixels × 32 apertures. This shows
Python scalar processing dominates the measured reference. It does not distinguish
every expression's cost; no memory-bandwidth bottleneck claim is made. The experiment
therefore targets interpreter overhead and scalar dispatch first.

`requirements-measured.txt` records the local environment, while `pyproject.toml`
provides portable dependency constraints. CI is configured for Python 3.11/3.13 on
Linux/Windows; local passing tests do not imply the hosted CI has run.
