"""Portable NPZ sweeps with their acquisition geometry and radar metadata."""
from dataclasses import asdict
import json
from pathlib import Path
import numpy as np
from .config import RadarConfig


def save_data(path: str | Path, data: np.ndarray, positions: np.ndarray, radar: RadarConfig) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, data=data, positions=positions, radar=json.dumps(asdict(radar)), schema=1)


def load_data(path: str | Path) -> tuple[np.ndarray, np.ndarray, RadarConfig]:
    try:
        loaded = np.load(path, allow_pickle=False)
    except (EOFError, OSError, ValueError) as exc:
        raise ValueError(f"cannot read NPZ data: {exc}") from exc

    try:
        with loaded as archive:
            required = {"schema", "data", "positions", "radar"}
            missing = required.difference(archive.files)
            if missing:
                raise ValueError(f"missing archive fields: {', '.join(sorted(missing))}")

            schema = np.asarray(archive["schema"])
            if schema.shape != () or not np.issubdtype(schema.dtype, np.integer) or schema.item() != 1:
                raise ValueError("unsupported data schema")

            data = np.asarray(archive["data"], dtype=np.complex128)
            positions = np.asarray(archive["positions"], dtype=float)
            radar = RadarConfig(**json.loads(str(archive["radar"])))
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        raise ValueError(f"invalid radar metadata: {exc}") from exc
    except (EOFError, OSError) as exc:
        raise ValueError(f"cannot read NPZ data: {exc}") from exc
    if data.ndim != 2 or data.shape[1] != radar.samples or positions.shape != (len(data), 2):
        raise ValueError("invalid sweep/trajectory dimensions")
    if len(data) == 0 or not np.isfinite(data).all() or not np.isfinite(positions).all():
        raise ValueError("data and positions must be nonempty and finite")
    return data, positions, radar


def write_json(path: str | Path, value: object) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
