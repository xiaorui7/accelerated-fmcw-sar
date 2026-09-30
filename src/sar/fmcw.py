"""Generate analytic dechirped baseband sweeps (not RF waveforms)."""
import numpy as np
from .config import C, RadarConfig
from .geometry import ranges


def simulate(scene: np.ndarray, positions: np.ndarray, radar: RadarConfig) -> np.ndarray:
    """s(t)=sum a exp(j 2pi [fc*tau + K*tau*t - K*tau**2/2])."""
    scene = np.asarray(scene, dtype=float)
    positions = np.asarray(positions, dtype=float)
    if scene.ndim != 2 or scene.shape[1] != 3 or len(scene) == 0 or not np.isfinite(scene).all():
        raise ValueError("scene must have finite (x,y,amplitude) rows")
    if positions.ndim != 2 or positions.shape[1] != 2 or len(positions) == 0 or not np.isfinite(positions).all():
        raise ValueError("positions must have finite (x,y) rows")
    tau = 2 * ranges(positions, scene[:, :2]) / C
    fs = radar.samples / radar.chirp_seconds
    if np.any(radar.slope * tau >= fs / 2):
        raise ValueError("scene beat frequency exceeds positive Nyquist range")
    t = np.arange(radar.samples) / fs
    data = np.zeros((len(positions), radar.samples), dtype=np.complex128)
    for index, amplitude in enumerate(scene[:, 2]):
        delay = tau[:, index, None]
        phase = radar.carrier_hz * delay + radar.slope * delay * t - radar.slope * delay**2 / 2
        data += amplitude * np.exp(2j * np.pi * phase)
    return data
