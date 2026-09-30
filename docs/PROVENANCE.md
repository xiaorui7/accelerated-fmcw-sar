# Source audit and attribution

At initial inspection on 2026-09-22, this workspace contained only `.git/`.
`git ls-tree HEAD` reported no valid HEAD: there was no committed source tree.
There were no MATLAB files, experiment scripts, figures, datasets, or generated
outputs to classify. The supplied attachment described the intended project;
it did not contain the undergraduate implementation.

Consequently **no old MATLAB implementation or old result was reused**. All Python
source, tests and scripts in this repository were newly authored for this milestone.
The 5-by-5 scene follows the requested approximate target count, but its coordinates
are illustrative and have not been checked against the 2024 experiment. Attribution
of original versus adapted undergraduate code remains pending receipt of those files.
Do not describe this repository as a verified port of that unavailable code.

| Current files | Classification |
|---|---|
| `src/sar/` | Newly implemented numerical pipeline and tooling |
| `tests/` | Newly implemented validation |
| `configs/`, `examples/`, `benchmarks/profile_reference.py` | New configurations and experiment scripts |
| `results/*.png`, `results/*.json`, `benchmarks/results.*`, `benchmarks/profile_reference.txt` | Generated on this machine |
| `docs/`, `README.md` | New explanations and evidence summaries |

No third-party source files were copied or vendored. Technical references informed
the separation of simulation, FFT range processing, BP and validation. Reference
repositories have their own licensing requirements; a future code import must audit
the license at the exact revision before copying anything.

## Reference review

- [Gorham & Moore, SAR Image Formation Toolbox for MATLAB (2010)](https://doi.org/10.1117/12.855375).
  The publisher page was inaccessible during this session. The linked
  [likemoongg/SARbackprojection README](https://github.com/likemoongg/SARbackprojection)
  identifies its basic BP and phase-history example as implementations based on that
  paper. This establishes a reference trail, not a claim that the complete paper was read.
- [Ulander, Hellsten & Stenström (2003)](https://research.chalmers.se/en/publication/6728),
  IEEE TAES 39(3), 760–776, DOI 10.1109/TAES.2003.1238734. Publication metadata was
  verified; a complete paper review was not available. FFBP remains outside this milestone.
- [IMS-AS-LUH/sar-sim](https://github.com/IMS-AS-LUH/sar-sim): README and `sarsim/simjob.py`
  were inspected. Distinct signal-generation, range-compression and azimuth-compression
  stages, trajectory distortion, and optional CUDA provide useful architectural context.
- [tbensonatl/gotcha-back](https://github.com/tbensonatl/gotcha-back): its README describes
  CPU reference reconstruction, multiple CUDA kernels, precision tradeoffs and SER
  comparisons. This project adopts reference-relative error reporting, but implements
  only CPU double-precision kernels and makes no GOTCHA performance comparison.
- [Ttl/fmcw3](https://github.com/Ttl/fmcw3): README, `process_sweeps.py` and
  `backprojection_tf.py` were inspected for the optional real-data investigation.
  Details and incompatibilities are recorded in [REAL_DATA.md](REAL_DATA.md).

These are sources for background and future work, not claimed implementations here.
