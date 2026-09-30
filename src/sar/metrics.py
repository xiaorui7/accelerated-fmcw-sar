"""Metrics with explicit reference dependence and no implicit image alignment."""
import numpy as np


def normalized_error(image: np.ndarray, reference: np.ndarray) -> float:
    if image.shape != reference.shape:
        raise ValueError("image shapes must match")
    norm = np.linalg.norm(reference)
    if norm == 0:
        raise ValueError("reference must be nonzero")
    return float(np.linalg.norm(image - reference) / norm)


def width_3db(axis: np.ndarray, magnitude: np.ndarray) -> float | None:
    """Interpolated contiguous half-power width; None for a clipped lobe."""
    peak = int(np.argmax(magnitude))
    threshold = magnitude[peak] / np.sqrt(2)
    if threshold <= 0:
        return None
    left = right = peak
    while left > 0 and magnitude[left] >= threshold:
        left -= 1
    while right < len(axis) - 1 and magnitude[right] >= threshold:
        right += 1
    if magnitude[left] >= threshold or magnitude[right] >= threshold:
        return None
    xl = np.interp(threshold, magnitude[left:left + 2], axis[left:left + 2])
    xr = np.interp(threshold, magnitude[right - 1:right + 1][::-1], axis[right - 1:right + 1][::-1])
    return float(xr - xl)


def point_metrics(image: np.ndarray, x: np.ndarray, y: np.ndarray,
                  target: tuple[float, float]) -> dict[str, float | None]:
    """Global-peak metrics intended only for isolated point-target scenes."""
    magnitude = np.abs(image)
    row, col = np.unravel_index(np.argmax(magnitude), image.shape)
    return {"localization_error_m": float(np.hypot(x[col] - target[0], y[row] - target[1])),
            "peak_magnitude": float(magnitude[row, col]),
            "width_3db_x_m": width_3db(x, magnitude[row]),
            "width_3db_y_m": width_3db(y, magnitude[:, col])}
