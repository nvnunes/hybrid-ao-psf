"""Independent response observations and native-free coma law calibration."""

import numpy as np
import pytest
from astropy import units as u

from hybrid_ao_psf import (
    CalibrationPhotometry,
    ComaCalibrationMeasurements,
    ComaCalibrationSettings,
    ComaResponseMeasurements,
    build_coma_calibration,
    load_coma_calibration,
    load_coma_measurements,
    save_coma_calibration,
    save_coma_measurements,
)


def measurement(angle):
    rng = np.random.default_rng(501 + angle)
    coma = rng.normal(size=(32, 2)) * 30 + 4
    magnitudes = np.arange(8.0, 21.0, 2.0)
    response = (0.1 + 0.04 * (1 / np.cos(np.deg2rad(angle)) - 1)) / (
        1 + 10 ** (0.4 * (magnitudes - 18.5))
    )
    tilt = coma[:, None, :] * response[None, :, None]
    noise = rng.normal(size=(32, len(magnitudes), 4, 2)) * 0.02
    noise -= noise.mean(axis=2, keepdims=True)
    return ComaCalibrationMeasurements(
        zenith_angle=angle * u.deg,
        response=ComaResponseMeasurements(
            coma=coma * u.nm,
            magnitudes=magnitudes * u.mag,
            tilt_mean=tilt * u.mas,
            tilt_batches=(tilt[:, :, None, :] + noise) * u.mas,
            tilt_noiseless=tilt * u.mas,
            dtype=np.float64,
        ),
        sensing_wavelength=710e-9 * u.m,
        photometry=CalibrationPhotometry(
            magnitude_band="fixture",
            photons_per_frame_zeropoint=1e8 * u.photon,
            dtype=np.float64,
        ),
        settings=ComaCalibrationSettings(samples=32, noise_draws=16),
        dtype=np.float64,
    )


def test_offline_coma_build_preserves_scalar_matrix_and_noise_diagnostics(tmp_path):
    planes = []
    for angle in (0, 20, 30, 40):
        original = measurement(angle)
        path = tmp_path / f"za{angle}.npz"
        save_coma_measurements(original, path)
        loaded = load_coma_measurements(path)
        np.testing.assert_array_equal(
            loaded.response.tilt_batches, original.response.tilt_batches
        )
        planes.append(loaded)
    result = build_coma_calibration(planes, dtype=np.float64)
    np.testing.assert_allclose(
        [result.model.a0.value, result.model.b.value, result.model.m_half.value],
        [0.1, 0.04, 18.5],
        rtol=1e-8,
        atol=1e-9,
    )
    for fit in result.diagnostics["planes"].values():
        np.testing.assert_allclose(
            np.asarray(fit["jacobian"]),
            np.asarray(fit["h"])[:, None, None] * np.eye(2),
            rtol=1e-12,
            atol=1e-12,
        )
        assert fit["diagnostics"]["bootstrapReplicates"] == 200
        assert fit["diagnostics"]["noiseAveraging"]["numberOfBatches"] == 4
    model_path = tmp_path / "model.json"
    save_coma_calibration(result.model, model_path)
    model = load_coma_calibration(model_path)
    magnitude = np.array([8, 18, 20]) * u.mag
    actual = model.predict(magnitude=magnitude, zenith_angle=20 * u.deg)
    expected = (0.1 + 0.04 * (1 / np.cos(np.deg2rad(20)) - 1)) / (
        1 + 10 ** (0.4 * (magnitude.value - 18.5))
    )
    np.testing.assert_allclose(actual.to_value(u.mas / u.nm), expected, rtol=1e-8)
    with pytest.raises(ValueError, match="support"):
        model.predict(magnitude=21 * u.mag, zenith_angle=20 * u.deg)
    with pytest.raises(ValueError, match="support"):
        model.predict(magnitude=10 * u.mag, zenith_angle=41 * u.deg)
    with pytest.raises(ValueError):
        model.a0.value.flags.writeable = True
    view = model.a0
    with pytest.raises(ValueError):
        view <<= u.arcsec / u.nm
    assert model.a0.unit == u.mas / u.nm


def test_measured_support_is_not_expanded():
    planes = [measurement(angle) for angle in (10, 20, 30)]
    result = build_coma_calibration(planes, heldout_zenith=None, dtype=np.float64)
    np.testing.assert_array_equal(result.model.zenith_support.value, [10, 30])
    with pytest.raises(ValueError, match="support"):
        result.model.predict(magnitude=10 * u.mag, zenith_angle=0 * u.deg)
    with pytest.raises(ValueError, match="three"):
        build_coma_calibration(planes[:2], dtype=np.float64)


def test_bootstrap_override_reuses_paired_observations():
    planes = [measurement(angle) for angle in (0, 20, 40)]
    result = build_coma_calibration(
        planes, bootstrap_seed=57, heldout_zenith=None, dtype=np.float64
    )
    assert result.diagnostics["bootstrap_seeds"] == (57, 57, 57)
    assert all(plane.settings.bootstrap_seed == 901732 for plane in planes)
    with pytest.raises(ValueError, match="bootstrap_seed"):
        build_coma_calibration(planes, bootstrap_seed=-1, dtype=np.float64)
