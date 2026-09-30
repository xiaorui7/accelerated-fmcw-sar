"""Composition of simulation, range processing, image formation and motion."""
import numpy as np
from .backprojection import BACKENDS
from .config import Config
from .fmcw import simulate
from .geometry import trajectory, image_grid
from .metrics import normalized_error, point_metrics
from .motion import perturb, correct
from .range_compression import range_compress, phase_coefficients
from .scene import make_scene


def reconstruct(config: Config, backend: str = "numba", data: np.ndarray | None = None,
                positions: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    positions = trajectory(config) if positions is None else positions
    data = simulate(make_scene(config), positions, config.radar) if data is None else data
    profiles, bins = range_compress(data, config.radar)
    x, y, pixels = image_grid(config.grid)
    image = BACKENDS[backend](profiles, bins, positions, pixels, phase_coefficients(config.radar))
    return image.reshape(len(y), len(x)), x, y


def motion_experiment(config: Config, backend: str = "numba") -> tuple[dict, dict, np.ndarray, np.ndarray]:
    nominal = trajectory(config)
    actual, error = perturb(nominal, config.jitter_m, config.offset_m, config.seed)
    data = simulate(make_scene(config), actual, config.radar)
    ideal, x, y = reconstruct(config, backend)
    uncorrected = reconstruct(config, backend, data, nominal)[0]
    corrected = reconstruct(config, backend, data, correct(nominal, error))[0]
    oracle = reconstruct(config, backend, data, actual)[0]
    images = {"Ideal acquisition": ideal, "Motion / nominal geometry": uncorrected,
              "Known-error correction": corrected}
    metrics = {"uncorrected_nrmse_to_actual_geometry": normalized_error(uncorrected, oracle),
               "corrected_nrmse_to_actual_geometry": normalized_error(corrected, oracle),
               "corrected_nrmse_to_ideal_acquisition": normalized_error(corrected, ideal)}
    if config.scene == "single":
        metrics["point_metrics"] = {name: point_metrics(im, x, y, (0, 10)) for name, im in images.items()}
    return images, metrics, x, y
