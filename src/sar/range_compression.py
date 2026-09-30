"""Windowed FFT and removal of the window-centre interpolation phase ramp."""
import numpy as np
from scipy.fft import fft, fftfreq
from .config import C, RadarConfig


def range_compress(data: np.ndarray, radar: RadarConfig) -> tuple[np.ndarray, np.ndarray]:
    if data.ndim != 2 or data.shape[1] != radar.samples or not np.isfinite(data).all():
        raise ValueError("data must be finite with shape (pulses, radar.samples)")
    window = np.hanning(radar.samples)
    nfft = radar.samples * radar.zero_pad
    fs = radar.samples / radar.chirp_seconds
    freq = fftfreq(nfft, 1 / fs)[:nfft // 2]
    profiles = fft(data * window, n=nfft, axis=1)[:, :nfft // 2] / window.sum()
    centre_time = (radar.samples - 1) / (2 * fs)
    profiles *= np.exp(2j * np.pi * freq * centre_time)
    return np.ascontiguousarray(profiles), C * freq / (2 * radar.slope)


def phase_coefficients(radar: RadarConfig) -> tuple[float, float]:
    """Phase at an interpolated range is k1*R + k2*R**2."""
    centre_time = (radar.samples - 1) * radar.chirp_seconds / (2 * radar.samples)
    return (4 * np.pi * (radar.carrier_hz + radar.slope * centre_time) / C,
            -4 * np.pi * radar.slope / C**2)
