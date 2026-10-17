"""Independent recurrence, pooled covariance, and storage precision evidence."""

import numpy as np
import pytest
from astropy import units as u

from hybrid_ao_psf import (
    CovarianceCalibrationSettings,
    CovarianceTelemetry,
    FixedGainIntegrator,
    build_covariance_calibration,
    load_covariance_calibration,
    load_covariance_measurements,
    measure_covariance_calibration,
    save_covariance_calibration,
    save_covariance_measurements,
)


def _source(dtype):
    rng = np.random.default_rng(119)
    latent = rng.normal(size=(4, 700, 4, 2))
    for k in range(1, 700):
        latent[:, k] += 0.8 * latent[:, k - 1]
    latent += np.arange(4)[:, None, None, None] * 3
    return CovarianceTelemetry(
        dtype=dtype,
        zenith_angle=20 * u.deg,
        frame_rate=500 * u.Hz,
        positions=[[0, 0], [1, 0], [0, 1], [1, 1]] * u.arcsec,
        tilt=latent * u.mas,
        coma=(latent[..., ::-1] * 2 + 0.2 * latent) * u.nm,
    )


def _reference(values, b, gain, delay):
    values = values.astype(np.float64)
    output = np.broadcast_to(values[:, :1] / b, values.shape).copy()
    for k in range(delay, values.shape[1]):
        output[:, k] = (
            output[:, k - 1]
            - gain * b * output[:, k - delay]
            + gain * values[:, k - delay]
        )
    return output


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_filtered_pooled_covariance_bank_matches_independent_reference(tmp_path, dtype):
    source = _source(dtype)
    controller = FixedGainIntegrator(
        dtype=dtype, gain=0.3, delay=2, frame_rate=500 * u.Hz
    )
    settings = CovarianceCalibrationSettings(
        feedback=(0.2, 1.0, 2.0), bootstrap_draws=7
    )
    measured = measure_covariance_calibration(
        source, controller, settings=settings, dtype=dtype
    )
    assert all(a.dtype == dtype for a in measured.tables.values())
    # One independent pooled mean, including deliberately different realization means.
    a = _reference(source.tilt.value, 0.2, 0.3, 2)[:, measured.discard :]
    b = _reference(source.coma.value, 2.0, 0.3, 2)[:, measured.discard :]
    a = a.transpose(2, 3, 0, 1).reshape(8, -1)
    b = b.transpose(2, 3, 0, 1).reshape(8, -1)
    expected = (
        (a - a.mean(-1, keepdims=True))
        @ (b - b.mean(-1, keepdims=True)).T
        / a.shape[-1]
    )
    expected = expected.reshape(4, 2, 4, 2).transpose(0, 2, 1, 3)
    tolerance = 256 * np.finfo(dtype).eps
    scale = np.max(np.abs(expected))
    np.testing.assert_allclose(
        measured.tables["C_Tt_Tc"].value[:, :, 0, -1],
        expected,
        rtol=tolerance,
        atol=tolerance * scale,
    )
    path = tmp_path / "measurements.npz"
    save_covariance_measurements(measured, path)
    restored = load_covariance_measurements(path)
    assert restored.dtype == dtype
    for key in measured.tables:
        np.testing.assert_array_equal(restored.tables[key], measured.tables[key])
    model = build_covariance_calibration([restored], dtype=dtype).model
    path = tmp_path / "model.json"
    save_covariance_calibration(model, path)
    loaded = load_covariance_calibration(path)
    assert loaded.dtype == loaded.producer_dtype == dtype
    result = loaded.C_Tt_Tc(
        0.5 * u.arcsec,
        0.5 * u.arcsec,
        0.5 * u.arcsec,
        0.5 * u.arcsec,
        0.2,
        2.0,
        zenith_angle=20 * u.deg,
    )
    assert result.dtype == dtype
    # Bilinear interpolation in each field averages all 16 source pairs.
    np.testing.assert_allclose(
        result.value, expected.mean((0, 1)), rtol=tolerance, atol=tolerance * scale
    )
    promoted = loaded.prepare(dtype=np.float64)
    assert promoted.dtype == np.float64
    assert promoted.producer_dtype == dtype
    assert loaded.dtype == dtype
    assert not np.shares_memory(
        promoted._bank.arrays["C_t_t"], loaded._bank.arrays["C_t_t"]
    )


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_known_wfs_response_and_population_noise_in_selected_precision(dtype):
    from hybrid_ao_psf import SensingWfsMeasurements, estimate_sensing_statistics

    rng = np.random.default_rng(324)
    zero = rng.normal(size=(4096, 2)) @ np.array([[2.0, 0.3], [0.1, 1.0]]) + [
        31.0,
        -23.0,
    ]
    matrix = np.array([[0.8, 0.05], [-0.03, 1.2]])
    paired = rng.normal(size=(1024, 2))
    shifted = np.stack(
        [
            paired + 10 * matrix[:, 0],
            paired - 10 * matrix[:, 0],
            paired + 10 * matrix[:, 1],
            paired - 10 * matrix[:, 1],
        ]
    )
    samples = SensingWfsMeasurements(
        dtype=dtype, zero=zero * u.mas, shifted=shifted * u.mas
    )
    stats = estimate_sensing_statistics(
        samples, displacement=10 * u.mas, block_draws=64, dtype=dtype
    )
    for name in (
        "mean",
        "covariance",
        "white",
        "gain_matrix",
        "gain_standard_error",
        "gain_blocks",
    ):
        assert getattr(stats, name).dtype == dtype
    tolerance = 128 * np.finfo(dtype).eps
    np.testing.assert_allclose(
        stats.gain_matrix.value, matrix, rtol=tolerance, atol=tolerance
    )
    reference = samples.zero.value.astype(np.float64)
    np.testing.assert_allclose(
        stats.covariance.value,
        np.cov(reference.T, ddof=0),
        rtol=tolerance,
        atol=tolerance,
    )
    np.testing.assert_allclose(
        stats.white.value.T @ stats.white.value / len(reference),
        np.eye(2),
        rtol=tolerance,
        atol=tolerance,
    )


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_coma_fit_precision_roundtrip_and_runtime_preparation(tmp_path, dtype):
    from hybrid_ao_psf import (
        build_coma_calibration,
        load_coma_calibration,
        save_coma_calibration,
    )
    from tests.test_coma_calibration import measurement

    planes = [measurement(angle) for angle in (0, 20, 30, 40)]
    result = build_coma_calibration(planes, dtype=dtype)
    model = result.model
    np.testing.assert_allclose(
        [model.a0.value, model.b.value, model.m_half.value],
        [0.1, 0.04, 18.5],
        rtol=max(1e-8, 128 * np.finfo(dtype).eps),
        atol=1e-9,
    )
    path = tmp_path / "model.json"
    save_coma_calibration(model, path)
    loaded = load_coma_calibration(path)
    assert loaded.dtype == np.dtype(dtype)
    predicted = loaded.predict(magnitude=[10, 15, 18] * u.mag, zenith_angle=20 * u.deg)
    assert predicted.dtype == dtype
    promoted = loaded.prepare(dtype=np.float64)
    assert promoted.dtype == np.float64
    assert np.dtype(promoted.producer_dtype) == dtype
    np.testing.assert_array_equal(promoted.a0.value, loaded.a0.value)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_noise_blur_analytic_derivatives_and_fit(dtype):
    from hybrid_ao_psf import NoiseBlurModel
    from hybrid_ao_psf._sensing_statistics import (
        ReplayCase,
        ReplayStatistics,
        _bin_probability,
        _bin_probability_gradient,
        fit_noise_blur,
    )

    # Differentiate independent perturbed bin evaluations in double precision.
    for values in (
        [0.5, 0.8, 0.95],
        [0.95, 0.2, 0.7],
        [0.011, 0.011, 0.999],
        [0.989, 0.99, 0.98],
    ):
        model = NoiseBlurModel(*values, dtype=dtype)
        for count in (1, 2):
            gradient = _bin_probability_gradient(model, count)
            assert gradient.dtype == np.dtype(dtype)
            for coordinate, column in ((0, 0), (count, 1)):
                step = np.zeros(3)
                step[coordinate] = 1e-6
                reference = (
                    _bin_probability(
                        NoiseBlurModel(*(np.array(values) + step), dtype=np.float64),
                        count,
                    )
                    - _bin_probability(
                        NoiseBlurModel(*(np.array(values) - step), dtype=np.float64),
                        count,
                    )
                ) / 2e-6
                np.testing.assert_allclose(
                    gradient[:, column],
                    reference,
                    atol=2e-7 if dtype == np.float32 else 1e-9,
                    rtol=0,
                )
    truth = NoiseBlurModel(0.7, 0.4, 0.8, dtype=np.float64)
    cases = [
        ReplayCase(
            str(n),
            n,
            "synthetic",
            ReplayStatistics(
                _bin_probability(truth, n).astype(dtype),
                np.zeros((4, 257), dtype=dtype),
                0.0,
                np.eye(2, dtype=dtype),
                1000,
            ),
        )
        for n in (1, 2)
    ]
    fitted, _ = fit_noise_blur(cases, dtype=dtype)
    radii = np.geomspace(1e-4, 100, 2048)
    for n in (1, 2):
        np.testing.assert_allclose(
            fitted.radial_cdf(radii, n),
            truth.radial_cdf(radii, n),
            atol=2e-4 if dtype == np.float32 else 1e-7,
            rtol=0,
        )
