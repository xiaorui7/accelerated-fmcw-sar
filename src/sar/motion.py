"""Known navigation-error experiments; no estimated autofocus."""
import numpy as np


def perturb(positions: np.ndarray, jitter_m: float, offset_m: tuple[float, float],
            seed: int) -> tuple[np.ndarray, np.ndarray]:
    if not np.isfinite(jitter_m) or jitter_m < 0:
        raise ValueError("jitter_m must be finite and nonnegative")
    offset = np.asarray(offset_m, dtype=float)
    if offset.shape != (2,) or not np.isfinite(offset).all():
        raise ValueError("offset must have two finite entries")
    error = np.random.default_rng(seed).normal(0, jitter_m, positions.shape) + offset
    return positions + error, error


def correct(nominal: np.ndarray, known_error: np.ndarray) -> np.ndarray:
    """Recover the actual acquisition positions from nominal + known error."""
    if nominal.shape != known_error.shape:
        raise ValueError("trajectory and error shapes must match")
    return nominal + known_error
