from dataclasses import replace
import numpy as np
import numba
import pytest
from sar.backprojection import bp_reference, bp_vectorized, bp_numba
from sar.config import C, Config, GridConfig, RadarConfig, load_config
from sar.fmcw import simulate
from sar.geometry import ranges, trajectory, image_grid
from sar.io import save_data, load_data
from sar.metrics import normalized_error, width_3db, point_metrics
from sar.pipeline import reconstruct, motion_experiment
from sar.range_compression import range_compress, phase_coefficients
from sar.scene import make_scene

numba.set_num_threads(min(4, numba.config.NUMBA_NUM_THREADS))


@pytest.fixture
def config():
    return Config(scene="single", pulses=32, grid=GridConfig(-.5, .5, 9.5, 10.5, 41))


def test_geometry():
    np.testing.assert_allclose(ranges(np.array([[0, 0], [3, 0]]), np.array([[3, 4], [0, 0]])),
                               [[5, 0], [4, 3]])


def test_single_target_localizes(config):
    image, x, y = reconstruct(config, "reference")
    metrics = point_metrics(image, x, y, (0, 10))
    assert metrics["localization_error_m"] <= (x[1] - x[0])
    assert metrics["peak_magnitude"] > .98
    assert 0 < metrics["width_3db_x_m"] < .5
    assert 0 < metrics["width_3db_y_m"] < .5


def test_multitarget_deterministic(config):
    config = replace(config, scene="point_targets")
    scene = make_scene(config)
    assert scene.shape == (25, 3)
    np.testing.assert_array_equal(simulate(scene, trajectory(config), config.radar),
                                  simulate(scene, trajectory(config), config.radar))


@pytest.mark.parametrize("backend", [bp_vectorized, bp_numba])
def test_backend_agreement(config, backend):
    positions = trajectory(config)
    profiles, bins = range_compress(simulate(make_scene(config), positions, config.radar), config.radar)
    args = profiles, bins, positions, image_grid(config.grid)[2], phase_coefficients(config.radar)
    expected = bp_reference(*args)
    actual = backend(*args)
    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-11)
    assert normalized_error(actual, expected) < 1e-10


@pytest.mark.parametrize("backend", [bp_reference, bp_vectorized, bp_numba])
def test_interpolation_edges_and_phase(backend):
    profiles = np.array([[1 + 2j, 3 + 4j, 5 + 6j]])
    bins = np.array([1., 2., 3.])
    positions = np.array([[0., 0.]])
    pixels = np.array([[0, .5], [0, 1], [0, 1.5], [0, 3], [0, 4.]])
    expected = np.array([0j, 1 + 2j, 2 + 3j, 5 + 6j, 0j]) * np.exp(-.3j * pixels[:, 1])
    np.testing.assert_allclose(backend(profiles, bins, positions, pixels, (.3, 0)), expected, atol=1e-12)


def test_fft_range_and_phase_analytic():
    radar = RadarConfig()
    # Exactly FFT-aligned target gives unit amplitude and independently predictable phase.
    distance = 64 * C / (2 * radar.bandwidth_hz)
    scene = np.array([[0, distance, 1]])
    positions = np.array([[0., 0.]])
    data = simulate(scene, positions, radar)
    tau = 2 * distance / C
    assert data[0, 0] == pytest.approx(np.exp(2j * np.pi * (radar.carrier_hz * tau - radar.slope * tau**2 / 2)))
    profiles, bins = range_compress(data, radar)
    peak = int(np.argmax(np.abs(profiles[0])))
    assert bins[peak] == pytest.approx(distance)
    k1, k2 = phase_coefficients(radar)
    assert profiles[0, peak] * np.exp(-1j * (k1 * distance + k2 * distance**2)) == pytest.approx(1, abs=1e-10)


def test_off_bin_matches_direct_matched_filter():
    radar = RadarConfig()
    positions = np.array([[-.123, .04], [.2, -.04]])
    scene = np.array([[.17, 10.013, .7]])
    data = simulate(scene, positions, radar)
    profiles, bins = range_compress(data, radar)
    value = bp_reference(profiles, bins, positions, scene[:, :2], phase_coefficients(radar))[0]
    # At the exact target, the independently generated matched template cancels every sample phase.
    template = simulate(np.array([[.17, 10.013, 1.]]), positions, radar)
    direct = np.mean(np.sum(data * template.conj() * np.hanning(radar.samples), axis=1) / np.hanning(radar.samples).sum())
    assert abs(value - direct) < .003


def test_motion_degrades_and_known_correction_restores(config):
    images, metrics, _, _ = motion_experiment(config)
    assert metrics["uncorrected_nrmse_to_actual_geometry"] > .3
    assert metrics["corrected_nrmse_to_actual_geometry"] < 1e-12
    assert np.abs(images["Known-error correction"]).max() > np.abs(images["Motion / nominal geometry"]).max()


def test_io_roundtrip(config, tmp_path):
    positions = trajectory(config)
    data = simulate(make_scene(config), positions, config.radar)
    path = tmp_path / "test.npz"
    save_data(path, data, positions, config.radar)
    actual, actual_positions, radar = load_data(path)
    np.testing.assert_array_equal(data, actual)
    np.testing.assert_array_equal(positions, actual_positions)
    assert radar == config.radar


@pytest.mark.parametrize("text", ["unknown: 2", "pulses: 1", "grid: {size: 1}", "scene: custom",
                                         "jitter_m: -1", "radar: {samples: 3}", "[]", "seed: -1"])
def test_invalid_config(tmp_path, text):
    path = tmp_path / "invalid.yaml"
    path.write_text(text)
    with pytest.raises(ValueError):
        load_config(path)


def test_custom_config(tmp_path):
    path = tmp_path / "custom.yaml"
    path.write_text("scene: custom\ntargets: [[0, 10, 0.5]]\n")
    np.testing.assert_array_equal(make_scene(load_config(path)), [[0, 10, .5]])


def test_metrics():
    axis = np.linspace(-2, 2, 1001)
    assert width_3db(axis, np.exp(-axis**2)) == pytest.approx(np.sqrt(2 * np.log(2)), rel=1e-4)
    assert width_3db(axis, np.ones(len(axis))) is None
    with pytest.raises(ValueError):
        normalized_error(np.ones(2), np.zeros(2))


def test_aliasing_rejected():
    with pytest.raises(ValueError, match="Nyquist"):
        simulate(np.array([[0, 100, 1]]), np.zeros((1, 2)), RadarConfig())


def test_vectorized_chunk_size(config):
    pos = trajectory(config)
    profiles, bins = range_compress(simulate(make_scene(config), pos, config.radar), config.radar)
    pixels = image_grid(config.grid)[2][:13]
    args = profiles, bins, pos, pixels, phase_coefficients(config.radar)
    np.testing.assert_allclose(bp_vectorized(*args, chunk_size=3), bp_reference(*args), atol=1e-11)
    with pytest.raises(ValueError):
        bp_vectorized(*args, chunk_size=0)


@pytest.mark.parametrize("backend", [bp_vectorized, bp_numba])
def test_arbitrary_complex_profiles(backend):
    rng = np.random.default_rng(31)
    profiles = rng.normal(size=(7, 20)) + 1j * rng.normal(size=(7, 20))
    bins = np.linspace(1., 5., 20)
    positions = rng.normal(size=(7, 2))
    pixels = rng.normal(size=(23, 2)) * 3
    args = profiles, bins, positions, pixels, (200., -.015)
    np.testing.assert_allclose(backend(*args), bp_reference(*args), rtol=1e-10, atol=1e-11)
