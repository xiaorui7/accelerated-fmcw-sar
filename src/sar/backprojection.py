"""Consistent scalar, chunked NumPy, and parallel Numba backprojectors."""
import math
import numpy as np
from numba import njit, prange


def _validate(profiles: np.ndarray, bins: np.ndarray, positions: np.ndarray,
              pixels: np.ndarray, phase: tuple[float, float]) -> None:
    if profiles.ndim != 2 or positions.shape != (profiles.shape[0], 2) or profiles.shape[0] == 0:
        raise ValueError("profiles and positions must have matching nonempty pulse axes")
    if pixels.ndim != 2 or pixels.shape[1] != 2:
        raise ValueError("pixels must have shape (n,2)")
    if bins.ndim != 1 or len(bins) < 2 or profiles.shape[1] != len(bins):
        raise ValueError("range bins must match profiles")
    if len(phase) != 2:
        raise ValueError("phase must contain linear and quadratic coefficients")
    if not all(np.isfinite(a).all() for a in (profiles, bins, positions, pixels, phase)):
        raise ValueError("all inputs must be finite")
    if not np.allclose(np.diff(bins), bins[1] - bins[0], rtol=1e-10, atol=1e-14) or bins[1] <= bins[0]:
        raise ValueError("range bins must be uniformly increasing")


def bp_reference(profiles: np.ndarray, bins: np.ndarray, positions: np.ndarray,
                 pixels: np.ndarray, phase: tuple[float, float]) -> np.ndarray:
    """Scalar pixel/pulse loops, complex linear interpolation, mean accumulation."""
    _validate(profiles, bins, positions, pixels, phase)
    result = np.zeros(len(pixels), dtype=np.complex128)
    step = bins[1] - bins[0]
    for p, (x, y) in enumerate(pixels):
        total = 0j
        for a, (ax, ay) in enumerate(positions):
            distance = math.sqrt((x - ax)**2 + (y - ay)**2)
            u = (distance - bins[0]) / step
            if 0 <= u <= len(bins) - 1:
                lower = min(int(u), len(bins) - 2)
                fraction = u - lower
                value = profiles[a, lower] * (1 - fraction) + profiles[a, lower + 1] * fraction
                angle = phase[0] * distance + phase[1] * distance**2
                total += value * complex(math.cos(angle), -math.sin(angle))
        result[p] = total / len(positions)
    return result


def bp_vectorized(profiles: np.ndarray, bins: np.ndarray, positions: np.ndarray,
                  pixels: np.ndarray, phase: tuple[float, float], chunk_size: int = 8192) -> np.ndarray:
    """Vectorize across bounded pixel chunks; avoid a pulses-by-pixels tensor."""
    _validate(profiles, bins, positions, pixels, phase)
    if type(chunk_size) is not int or chunk_size < 1:
        raise ValueError("chunk_size must be a positive integer")
    result = np.zeros(len(pixels), dtype=np.complex128)
    step = bins[1] - bins[0]
    for start in range(0, len(pixels), chunk_size):
        chunk = pixels[start:start + chunk_size]
        total = np.zeros(len(chunk), dtype=np.complex128)
        for a, position in enumerate(positions):
            delta = chunk - position
            distance = np.sqrt(delta[:, 0]**2 + delta[:, 1]**2)
            u = (distance - bins[0]) / step
            valid = (u >= 0) & (u <= len(bins) - 1)
            clipped = np.clip(u, 0, len(bins) - 1)
            lower = np.minimum(clipped.astype(np.int64), len(bins) - 2)
            fraction = clipped - lower
            value = profiles[a, lower] * (1 - fraction) + profiles[a, lower + 1] * fraction
            angle = phase[0] * distance + phase[1] * distance**2
            total += np.where(valid, value * np.exp(-1j * angle), 0)
        result[start:start + len(chunk)] = total / len(positions)
    return result


@njit(cache=True, parallel=True)
def _numba_kernel(profiles: np.ndarray, bins: np.ndarray, positions: np.ndarray,
                  pixels: np.ndarray, k1: float, k2: float) -> np.ndarray:
    result = np.zeros(len(pixels), dtype=np.complex128)
    step = bins[1] - bins[0]
    for p in prange(len(pixels)):
        total = 0j
        for a in range(len(positions)):
            dx = pixels[p, 0] - positions[a, 0]
            dy = pixels[p, 1] - positions[a, 1]
            distance = math.sqrt(dx * dx + dy * dy)
            u = (distance - bins[0]) / step
            if 0 <= u <= len(bins) - 1:
                lower = min(int(u), len(bins) - 2)
                fraction = u - lower
                value = profiles[a, lower] * (1 - fraction) + profiles[a, lower + 1] * fraction
                angle = k1 * distance + k2 * distance * distance
                total += value * complex(math.cos(angle), -math.sin(angle))
        result[p] = total / len(positions)
    return result


def bp_numba(profiles: np.ndarray, bins: np.ndarray, positions: np.ndarray,
             pixels: np.ndarray, phase: tuple[float, float]) -> np.ndarray:
    """Parallel float64/complex128 kernel; no fastmath or pulse reduction races."""
    _validate(profiles, bins, positions, pixels, phase)
    return _numba_kernel(profiles, bins, positions, pixels, phase[0], phase[1])


BACKENDS = {"reference": bp_reference, "vectorized": bp_vectorized, "numba": bp_numba}
