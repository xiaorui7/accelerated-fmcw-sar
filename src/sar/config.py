"""Validated SI-unit configuration; unknown keys are rejected."""
from dataclasses import dataclass, field
from pathlib import Path
import math
import yaml

C = 299_792_458.0


@dataclass(frozen=True)
class RadarConfig:
    carrier_hz: float = 6e9
    bandwidth_hz: float = 1e9
    chirp_seconds: float = 100e-6
    samples: int = 256
    zero_pad: int = 8

    def __post_init__(self) -> None:
        for name in ("carrier_hz", "bandwidth_hz", "chirp_seconds"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name, minimum in (("samples", 8), ("zero_pad", 1)):
            value = getattr(self, name)
            if type(value) is not int or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")

    @property
    def slope(self) -> float:
        return self.bandwidth_hz / self.chirp_seconds


@dataclass(frozen=True)
class GridConfig:
    x_min: float = -2.0
    x_max: float = 2.0
    y_min: float = 8.0
    y_max: float = 12.0
    size: int = 128

    def __post_init__(self) -> None:
        if not all(math.isfinite(v) for v in (self.x_min, self.x_max, self.y_min, self.y_max)):
            raise ValueError("grid limits must be finite")
        if self.x_min >= self.x_max or self.y_min >= self.y_max:
            raise ValueError("grid limits must be increasing")
        if type(self.size) is not int or self.size < 2:
            raise ValueError("grid size must be an integer >= 2")


@dataclass(frozen=True)
class Config:
    radar: RadarConfig = field(default_factory=RadarConfig)
    grid: GridConfig = field(default_factory=GridConfig)
    pulses: int = 64
    aperture_m: float = 2.0
    scene: str = "point_targets"
    targets: tuple[tuple[float, float, float], ...] = ()
    seed: int = 2024
    jitter_m: float = 0.008
    offset_m: tuple[float, float] = (0.0, 0.005)

    def __post_init__(self) -> None:
        if type(self.pulses) is not int or self.pulses < 2:
            raise ValueError("pulses must be an integer >= 2")
        if not math.isfinite(self.aperture_m) or self.aperture_m <= 0:
            raise ValueError("aperture_m must be finite and positive")
        if not math.isfinite(self.jitter_m) or self.jitter_m < 0:
            raise ValueError("jitter_m must be finite and nonnegative")
        if len(self.offset_m) != 2 or not all(math.isfinite(v) for v in self.offset_m):
            raise ValueError("offset_m must contain two finite coordinates")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        if self.scene not in ("single", "point_targets", "custom"):
            raise ValueError("unknown scene")
        if self.scene == "custom" and not self.targets:
            raise ValueError("custom scene requires targets")
        for target in self.targets:
            if len(target) != 3 or not all(math.isfinite(v) for v in target):
                raise ValueError("targets must be finite [x, y, amplitude] triples")


def load_config(path: str | Path) -> Config:
    """Load and validate a YAML configuration file."""
    try:
        with Path(path).open(encoding="utf-8") as stream:
            raw = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        raise ValueError(f"invalid YAML: {exc}") from exc
    if not isinstance(raw, dict):
        raise ValueError("configuration must be a mapping")
    try:
        raw["radar"] = RadarConfig(**raw.get("radar", {}))
        raw["grid"] = GridConfig(**raw.get("grid", {}))
        return Config(**raw)
    except TypeError as exc:
        raise ValueError(f"invalid configuration fields: {exc}") from exc
