# Resume Entry and Interview Preparation

## Resume entry

**High-Performance Telemetry Processing Engine** — Python, NumPy, Numba, PyYAML, pytest, GitHub Actions

- Built a validated processing engine for heterogeneous industrial telemetry, aligning multi-rate pressure, flow, temperature, vibration, and power channels into configurable per-cycle numerical features.
- Designed interchangeable Python, NumPy, and parallel Numba backends, reaching 21,789 cycles/second and 27.34× reference throughput across 1,000 cycles with verified numerical agreement.
- Implemented bounded-memory batching with 10.4 MiB lower peak RSS at batch 32 versus 256, backed by 41 tests, 78% coverage, correctness guards, and cross-platform CI.

For a broader SDE audience, bullet 2 can instead use the system-level result:

- Benchmarked the complete file-to-output pipeline at 391.5 cycles/second across 1,000 multi-rate cycles, including loading, validation, alignment, parallel feature extraction, diagnostic rules, and structured output.

## Interview questions

1. **Why did you create a reference backend?**  
   It is an intentionally readable oracle. The NumPy and Numba implementations can be
   optimized aggressively while tests and benchmarks still compare every feature with a
   simple implementation whose loops and quantile definitions are easy to inspect.

2. **Why is NumPy faster than normal Python here?**  
   NumPy moves reductions over contiguous float64 arrays into compiled native loops. It
   avoids Python bytecode dispatch and object handling for each of the three million
   samples in the saved benchmark.

3. **What does Numba JIT do?**  
   Numba specializes the feature kernel for the concrete array types and compiles the
   loop to machine code. Compilation costs about the recorded warm-up time, so the
   steady-state benchmark warms it first and never calls that one-time cost application
   latency.

4. **How did you parallelize the workload safely?**  
   The kernel flattens independent cycle/channel pairs and applies `prange`. Each
   iteration uses local sums and owns one output feature row, so no two threads mutate
   the same memory and no shared reduction is required.

5. **How did you verify optimized results were correct?**  
   Hand-computed feature tests validate the definitions. Random tensors then compare
   NumPy and Numba with the reference at `1e-12` tolerances. Benchmark workers also fail
   above `1e-10`; measured maxima were `4.62e-14` and `2.22e-16`.

6. **Why use batch processing?**  
   Loading every cycle would make memory proportional to dataset size. The source
   iterator and batch loop retain only a configured number of cycles, write results as
   they are produced, and release each batch before reading the next one.

7. **What trade-off exists between batch size and memory?**  
   Larger batches reduce orchestration overhead but increase the live input and scratch
   arrays. On this machine, batch 64 was fastest, while batch 32 used 10.4 MiB less peak
   RSS than batch 256. The best choice depends on workload and hardware.

8. **How did you handle sensors with different sampling rates?**  
   Every channel retains its own rate. Alignment builds a configurable common relative-
   time grid and applies linear or nearest interpolation inside the common duration.
   Documentation states that downsampling can discard high-frequency information and
   that interpolation cannot recreate unmeasured data.
