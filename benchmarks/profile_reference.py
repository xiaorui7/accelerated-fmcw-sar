"""Run before introducing optimized kernels; saves cumulative cProfile output."""
import cProfile
from pathlib import Path
import pstats
from sar.config import Config, GridConfig
from sar.scene import make_scene
from sar.geometry import trajectory, image_grid
from sar.fmcw import simulate
from sar.range_compression import range_compress, phase_coefficients
from sar.backprojection import bp_reference


def main() -> None:
    config = Config(grid=GridConfig(size=64), pulses=32, scene="single")
    positions = trajectory(config)
    profiles, bins = range_compress(simulate(make_scene(config), positions, config.radar), config.radar)
    pixels = image_grid(config.grid)[2]
    profiler = cProfile.Profile()
    profiler.runcall(bp_reference, profiles, bins, positions, pixels, phase_coefficients(config.radar))
    Path("benchmarks").mkdir(exist_ok=True)
    with open("benchmarks/profile_reference.txt", "w", encoding="utf-8") as stream:
        pstats.Stats(profiler, stream=stream).strip_dirs().sort_stats("cumulative").print_stats(20)


if __name__ == "__main__":
    main()
