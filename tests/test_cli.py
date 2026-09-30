import json
import pytest
from sar.cli import main


def test_cli_roundtrip_and_motion(tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text("scene: single\npulses: 8\ngrid: {size: 16}\n")
    common = ["--config", str(config), "--output", str(tmp_path)]
    main(["simulate", *common])
    main(["reconstruct", *common, "--backend", "vectorized", "--input", str(tmp_path / "sweeps.npz")])
    assert (tmp_path / "reconstruction.png").is_file()
    main(["motion-demo", *common, "--backend", "vectorized"])
    assert json.loads((tmp_path / "motion_metrics.json").read_text())["corrected_nrmse_to_actual_geometry"] == 0


def test_cli_bad_input(tmp_path):
    with pytest.raises(SystemExit) as error:
        main(["simulate", "--config", str(tmp_path / "missing.yaml")])
    assert error.value.code == 2


def test_cli_rejects_invalid_yaml(tmp_path):
    config = tmp_path / "invalid.yaml"
    config.write_text("radar: [")
    with pytest.raises(SystemExit) as error:
        main(["simulate", "--config", str(config)])
    assert error.value.code == 2
