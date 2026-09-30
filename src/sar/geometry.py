"""Planar monostatic stop-and-go geometry, in metres."""
import numpy as np
from .config import Config, GridConfig


def trajectory(config: Config) -> np.ndarray:
    return np.column_stack((np.linspace(-config.aperture_m / 2,
                                        config.aperture_m / 2, config.pulses),
                            np.zeros(config.pulses)))


def image_grid(config: GridConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.linspace(config.x_min, config.x_max, config.size)
    y = np.linspace(config.y_min, config.y_max, config.size)
    xx, yy = np.meshgrid(x, y)
    return x, y, np.column_stack((xx.ravel(), yy.ravel()))


def ranges(positions: np.ndarray, pixels: np.ndarray) -> np.ndarray:
    """Return pairwise Euclidean ranges, shape (apertures, pixels)."""
    return np.linalg.norm(positions[:, None, :] - pixels[None, :, :], axis=-1)
