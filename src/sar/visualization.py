"""Headless plotting kept outside numerical kernels."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def magnitude_db(image: np.ndarray, reference_peak: float | None = None, floor: float = -50) -> np.ndarray:
    peak = float(np.max(np.abs(image))) if reference_peak is None else reference_peak
    if peak <= 0:
        return np.full(image.shape, floor)
    return np.maximum(20 * np.log10(np.maximum(np.abs(image) / peak, 10**(floor / 20))), floor)


def plot_images(images: dict[str, np.ndarray], x: np.ndarray, y: np.ndarray,
                path: str | Path, targets: np.ndarray | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    peak = max(float(np.max(np.abs(im))) for im in images.values())
    fig, axes = plt.subplots(1, len(images), figsize=(5 * len(images), 4.6), squeeze=False,
                             layout="constrained")
    for ax, (title, im) in zip(axes[0], images.items()):
        artist = ax.imshow(magnitude_db(im, peak), origin="lower", extent=(x[0], x[-1], y[0], y[-1]),
                           vmin=-50, vmax=0, cmap="magma", aspect="equal")
        if targets is not None:
            ax.scatter(targets[:, 0], targets[:, 1], marker="+", color="cyan", s=25, linewidths=.8)
        ax.set(title=title, xlabel="Cross-range x (m)", ylabel="Down-range y (m)")
    fig.colorbar(artist, ax=axes[0].tolist(), label="Magnitude (dB, shared peak)", shrink=.8)
    fig.savefig(path, dpi=160)
    plt.close(fig)
