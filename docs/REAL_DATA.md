# Optional public-data investigation

The [Ttl/fmcw3 README](https://github.com/Ttl/fmcw3) links a
[parking-lot radar log](https://hforsten.com/fmcw3/parking_lot_sar.log) and recommends
a time slice of 20–136 seconds. No dataset was downloaded and no real-data
reconstruction is claimed in this milestone.

The inspected [sweep reader](https://github.com/Ttl/fmcw3/blob/master/pc/sar/process_sweeps.py)
reads a length-prefixed `fmcw3;` settings header, signed ADC samples and sweep framing,
then extracts selected time intervals into `(settings, ch1, ch2)` in a pickle file.
Metadata includes chirp start, bandwidth, sweep duration, delay and sample rate.

The inspected [BP script](https://github.com/Ttl/fmcw3/blob/master/pc/sar/backprojection_tf.py)
uses real-channel FFT processing, its own phase conjugation/RVP convention, a
transmit–receive offset, an assumed initial velocity, beam limits and TensorFlow
motion optimization. These are material differences from our analytic, monostatic,
known-position synthetic data. Merely relabelling that data as our NPZ format would
not establish physical correctness.

Recommended bounded next data milestone:

1. Check dataset redistribution terms and archive size; document a checksum and a
   small time interval without making CI download it.
2. Implement a streaming log reader with truncation/framing checks and unit tests on
   a hand-constructed tiny binary fixture. Do not require untrusted pickle loading.
3. Preserve actual sample rate independently of sweep duration and count, acquisition
   timing, channel identity and transmit–receive geometry.
4. Derive and test the phase-convention conversion against an analytic fixture before
   using the real data. Add calibrated geometry or explicitly label an assumed path.
5. Compare a small reconstruction against the upstream example and document residual
   geometric/calibration limitations. Only then claim a real-data demonstration.

The current `io.py` supports our own versioned NPZ schema with `allow_pickle=False`;
it is not an `fmcw3` adapter. The stretch-goal investigation is complete, but adapter
implementation and a validated real-data demonstration remain future work.
