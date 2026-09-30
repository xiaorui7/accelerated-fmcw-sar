"""Run from the repository root after editable installation."""
from sar.cli import main

if __name__ == "__main__":
    main(["reconstruct", "--config", "configs/point_targets.yaml"])
    main(["motion-demo", "--config", "configs/motion_error.yaml"])
