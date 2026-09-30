"""Deterministic point-scatterer scenes: columns x, y, amplitude (SI)."""
import numpy as np
from .config import Config


def make_scene(config: Config) -> np.ndarray:
    if config.scene == "custom":
        return np.asarray(config.targets, dtype=np.float64)
    if config.scene == "single":
        return np.array([[0.0, 10.0, 1.0]])
    x, y = np.meshgrid(np.linspace(-1.5, 1.5, 5), np.linspace(8.5, 11.5, 5))
    return np.column_stack((x.ravel(), y.ravel(), np.ones(25)))
