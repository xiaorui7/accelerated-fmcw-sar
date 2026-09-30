"""Run `python -m sar.cli --help` for CPU-only demonstrations."""
import argparse
from dataclasses import replace
import logging
from pathlib import Path
import numpy as np
import numba
from .backprojection import BACKENDS
from .config import load_config
from .fmcw import simulate
from .geometry import trajectory
from .io import save_data, load_data, write_json
from .pipeline import reconstruct, motion_experiment
from .scene import make_scene


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Accelerated FMCW-SAR imaging pipeline")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("simulate", "reconstruct", "motion-demo"):
        sub = commands.add_parser(name)
        sub.add_argument("--config", required=True)
        sub.add_argument("--output", type=Path, default=Path("results"))
        if name != "simulate":
            sub.add_argument("--backend", choices=BACKENDS, default="numba")
            sub.add_argument("--threads", type=int, default=4)
        if name == "reconstruct":
            sub.add_argument("--input", type=Path, help="NPZ sweep archive; its radar metadata overrides config radar")
    sub = commands.add_parser("benchmark")
    sub.add_argument("--output", type=Path, default=Path("benchmarks/results.csv"))
    sub.add_argument("--sizes", nargs="+", type=int, default=[64, 128, 256, 512])
    sub.add_argument("--pulses", nargs="+", type=int, default=[16, 64])
    sub.add_argument("--repeats", type=int, default=3)
    sub.add_argument("--threads", type=int, default=4)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        if hasattr(args, "threads"):
            if args.threads < 1:
                raise ValueError("threads must be positive")
            numba.set_num_threads(min(args.threads, numba.config.NUMBA_NUM_THREADS))
        if args.command == "benchmark":
            from .benchmark import run_benchmarks
            run_benchmarks(args.output, args.sizes, args.pulses, args.repeats, args.threads)
            return
        config = load_config(args.config)
        args.output.mkdir(parents=True, exist_ok=True)
        if args.command == "simulate":
            positions = trajectory(config)
            save_data(args.output / "sweeps.npz", simulate(make_scene(config), positions, config.radar),
                      positions, config.radar)
        else:
            from .visualization import plot_images
            if args.command == "reconstruct":
                data = positions = None
                if args.input:
                    data, positions, radar = load_data(args.input)
                    config = replace(config, radar=radar)
                image, x, y = reconstruct(config, args.backend, data, positions)
                np.savez_compressed(args.output / "reconstruction.npz", image=image, x=x, y=y)
                plot_images({f"BP ({args.backend})": image}, x, y, args.output / "reconstruction.png",
                            None if args.input else make_scene(config))
            else:
                images, metrics, x, y = motion_experiment(config, args.backend)
                plot_images(images, x, y, args.output / "motion.png")
                write_json(args.output / "motion_metrics.json", metrics)
                np.savez_compressed(args.output / "motion.npz", **images, x=x, y=y)
        logging.info("Saved outputs to %s", args.output)
    except (ValueError, OSError, KeyError) as exc:
        parser.exit(2, f"sar: {exc}\n")


if __name__ == "__main__":
    main()
