import numpy as np
import pytest

from sar.io import load_data


def test_load_data_rejects_missing_fields(tmp_path):
    path = tmp_path / "missing.npz"
    np.savez(path, schema=np.array(1))

    with pytest.raises(ValueError, match="missing archive fields"):
        load_data(path)


@pytest.mark.parametrize("schema", [np.array(2), np.array(1.0), np.array([1])])
def test_load_data_rejects_invalid_schema(tmp_path, schema):
    path = tmp_path / "schema.npz"
    np.savez(
        path,
        schema=schema,
        data=np.ones((1, 8), dtype=np.complex128),
        positions=np.zeros((1, 2)),
        radar='{"samples": 8}',
    )

    with pytest.raises(ValueError, match="unsupported data schema"):
        load_data(path)


def test_load_data_rejects_nonfinite_values(tmp_path):
    path = tmp_path / "nonfinite.npz"
    np.savez(
        path,
        schema=np.array(1),
        data=np.full((1, 8), np.nan, dtype=np.complex128),
        positions=np.zeros((1, 2)),
        radar='{"samples": 8}',
    )

    with pytest.raises(ValueError, match="nonempty and finite"):
        load_data(path)


def test_load_data_rejects_damaged_archive(tmp_path):
    path = tmp_path / "damaged.npz"
    path.write_bytes(b"not a zip archive")

    with pytest.raises(ValueError, match="cannot read NPZ data"):
        load_data(path)
