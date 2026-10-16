"""Independent pooled moments and physical six-function interpolation checks."""

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


def telemetry(zenith=20, realizations=3):
    rng = np.random.default_rng(770 + zenith)
    x, y = np.meshgrid([-1.0, 0.0, 1.0], [-1.0, 0.0, 1.0])
    positions = np.column_stack((x.ravel(), y.ravel()))[1:]
    latent = rng.normal(size=(realizations, 600, 5))
    for k in range(1, 600):
        latent[:, k] += 0.85 * latent[:, k - 1]
    latent += np.arange(realizations)[:, None, None] * 2
    weights = rng.normal(size=(len(positions), 2, 5))
    tilt = np.einsum("rfq,daq->rfda", latent, weights)
    coma = (
        np.einsum(
            "rfq,daq->rfda", latent, weights + 0.1 * rng.normal(size=weights.shape)
        )
        * 3
    )
    return CovarianceTelemetry(
        zenith_angle=zenith * u.deg,
        positions=positions * u.arcsec,
        frame_rate=500 * u.Hz,
        tilt=tilt * u.mas,
        coma=coma * u.nm,
    )


def direct_filter(history, b, gain, delay):
    output = np.broadcast_to(history[:, :1] / b, history.shape).copy()
    for k in range(delay, history.shape[1]):
        output[:, k] = (
            output[:, k - 1]
            - gain * b * output[:, k - delay]
            + gain * history[:, k - delay]
        )
    return output


def independent_tables(record, settings, cut):
    t = record.tilt.value[:, cut:]
    ft = np.stack(
        [
            direct_filter(record.tilt.value, b, 0.3, 2)[:, cut:]
            for b in settings.feedback
        ],
        axis=2,
    )
    fc = np.stack(
        [
            direct_filter(record.coma.value, b, 0.3, 2)[:, cut:]
            for b in settings.feedback
        ],
        axis=2,
    )
    t = t.transpose(2, 3, 0, 1).reshape(len(record.positions), 2, -1)
    ft = ft.transpose(3, 2, 4, 0, 1).reshape(
        len(record.positions), len(settings.feedback), 2, -1
    )
    fc = fc.transpose(3, 2, 4, 0, 1).reshape(ft.shape)
    t -= t.mean(-1, keepdims=True)
    ft -= ft.mean(-1, keepdims=True)
    fc -= fc.mean(-1, keepdims=True)
    n = t.shape[-1]
    return {
        "C_t_t": np.einsum("iak,jdk->ijad", t, t) / n,
        "C_Tc_Tc": np.einsum("ibak,jcdk->ijbcad", fc, fc) / n,
        "C_t_Tc": np.einsum("iak,jbdk->ijbad", t, fc) / n,
        "C_Tt_Tc": np.einsum("ibak,jcdk->ijbcad", ft, fc) / n,
        "C_Tt_Tt": np.einsum("ibak,jcdk->ijbcad", ft, ft) / n,
        "C_t_Tt": np.einsum("iak,jbdk->ijbad", t, ft) / n,
    }


@pytest.fixture
def measured():
    record = telemetry()
    controller = FixedGainIntegrator(gain=0.3, delay=2, frame_rate=500 * u.Hz)
    settings = CovarianceCalibrationSettings(
        feedback=(0.5, 1.0, 1.5), bootstrap_draws=11
    )
    return record, measure_covariance_calibration(record, controller, settings=settings)


def test_six_functions_match_independent_recurrence_and_pooled_mean(measured):
    record, result = measured
    expected = independent_tables(record, result.settings, result.discard)
    for name, values in expected.items():
        np.testing.assert_allclose(
            result.tables[name].value, values, rtol=1e-12, atol=1e-10
        )
    assert result.realizations == 3
    assert result.diagnostics["precision"]["bootstrap_samples"] == 11
    with pytest.raises(ValueError):
        result.tables["C_t_t"].value.flags.writeable = True
    with pytest.raises(ValueError):
        view = result.tables["C_t_t"]
        view *= 100 * u.one


def test_measurement_and_model_roundtrip_unequal_realizations(measured, tmp_path):
    record, first = measured
    other = measure_covariance_calibration(
        telemetry(40, 2), first.controller, settings=first.settings
    )
    path = tmp_path / "measurements.npz"
    save_covariance_measurements(first, path)
    loaded = load_covariance_measurements(path)
    np.testing.assert_array_equal(loaded.tables["C_t_Tc"], first.tables["C_t_Tc"])
    built = build_covariance_calibration([other, loaded])
    model_path = tmp_path / "model.json"
    save_covariance_calibration(built.model, model_path)
    model = load_covariance_calibration(model_path)
    np.testing.assert_array_equal(model.zenith_angles.value, [20, 40])
    args = [
        record.positions[0, 0],
        record.positions[0, 1],
        record.positions[1, 0],
        record.positions[1, 1],
    ]
    np.testing.assert_allclose(
        model.C_Tt_Tc(*args, 0.5, 1.5, zenith_angle=20 * u.deg).value,
        first.tables["C_Tt_Tc"][0, 1, 0, 2].value,
        rtol=1e-12,
        atol=1e-10,
    )
    with pytest.raises(ValueError, match="support"):
        model.C_t_Tt(*args, 0.1, zenith_angle=20 * u.deg)
    with pytest.raises(ValueError, match="support"):
        model.C_t_t(*args, zenith_angle=50 * u.deg)
    with pytest.raises(ValueError):
        model.feedback.flags.writeable = True
    with pytest.raises(TypeError):
        model.C_t_t(*[float(x.value) for x in args], zenith_angle=20 * u.deg)


def test_cadence_and_insufficient_settled_window_rejected():
    record = telemetry()
    with pytest.raises(ValueError, match="frame rates"):
        measure_covariance_calibration(
            record, FixedGainIntegrator(gain=0.3, delay=2, frame_rate=100 * u.Hz)
        )
    short = CovarianceTelemetry(
        zenith_angle=record.zenith_angle,
        positions=record.positions,
        frame_rate=record.frame_rate,
        tilt=record.tilt[:, :20],
        coma=record.coma[:, :20],
    )
    with pytest.raises(ValueError, match="insufficient frames"):
        measure_covariance_calibration(
            short, FixedGainIntegrator(gain=0.3, delay=2, frame_rate=500 * u.Hz)
        )
