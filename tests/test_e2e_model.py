"""Independent matrix replay and configured calibrated-model contracts."""

import builtins
import json
import subprocess
import sys
from dataclasses import replace

import numpy as np
import pytest
from astropy import units as u

from hybrid_ao_psf import (
    CalibrationPhotometry,
    ComaResponseModel,
    CovarianceCalibrationSettings,
    CovarianceTelemetry,
    E2ECalibratedModel,
    FixedGainIntegrator,
    NgsHoMetricSamples,
    NoiseBlurModel,
    ReconstructorNoiseModel,
    SensingCalibration,
    SensingCalibrationMeasurements,
    SensingWfsMeasurements,
    SensingWfsSampling,
    build_covariance_calibration,
    build_lo_reconstructor,
    build_ngs_ho_metric_interpolator,
    build_sensing_response,
    load_e2e_calibrated_model,
    load_sensing_calibration,
    measure_covariance_calibration,
    save_coma_calibration,
    save_covariance_calibration,
    save_ngs_ho_metric_interpolator,
    save_reconstructor_calibration,
    save_sensing_calibration,
)
from hybrid_ao_psf._e2e_model import _scalar_decomposition


def pooled(left, right):
    left, right = left.reshape(-1, 2), right.reshape(-1, 2)
    return (left - left.mean(0)).T @ (right - right.mean(0)) / len(left)


def matrix_replay(inputs, feedback, gain, delay):
    # Independent recurrence; no shared controller/filter/impulse helper.
    output = np.broadcast_to(
        np.linalg.solve(feedback, inputs[:, 0].T).T[:, None], inputs.shape
    ).copy()
    for k in range(delay, inputs.shape[1]):
        output[:, k] = (
            output[:, k - 1]
            - gain * output[:, k - delay] @ feedback.T
            + gain * inputs[:, k - delay]
        )
    return output


@pytest.fixture(scope="module")
def synthetic():
    rng = np.random.default_rng(7621)
    xy = np.array([(x, y) for x in (-1, 0, 1) for y in (-1, 0, 1)])
    latent = rng.normal(size=(5, 1200, 6))
    for k in range(1, latent.shape[1]):
        latent[:, k] += 0.8 * latent[:, k - 1]
    latent += np.arange(5)[:, None, None] * 0.8
    matrices = rng.normal(size=(1, 2, 6)) + 0.25 * rng.normal(size=(9, 2, 6))
    tilt = np.einsum("rfq,daq->rfda", latent, matrices)
    coma = np.einsum(
        "rfq,daq->rfda", latent, matrices + rng.normal(size=matrices.shape)
    )
    record = CovarianceTelemetry(
        zenith_angle=20 * u.deg,
        positions=xy * u.arcsec,
        frame_rate=500 * u.Hz,
        tilt=tilt * u.mas,
        coma=coma * u.nm,
        dtype=np.float64,
    )
    controller = FixedGainIntegrator(
        gain=0.3, delay=2, frame_rate=500 * u.Hz, dtype=np.float64
    )
    photometry = CalibrationPhotometry(
        magnitude_band="fixture",
        photons_per_frame_zeropoint=1e8 * u.photon,
        dtype=np.float64,
    )
    sampling = SensingWfsSampling(zero_draws=128, gain_draws=64, block_draws=8, seed=77)
    points = []
    for x, y in xy:
        for mag in (8, 10):
            gain = 0.8 + 0.08 * x + 0.04 * y
            noise = rng.normal(size=(128, 2))
            shifted = np.array([[10, 0], [-10, 0], [0, 10], [0, -10]])[
                :, None, :
            ] * gain + rng.normal(size=(1, 64, 2))
            points.append(
                SensingCalibrationMeasurements(
                    zenith_angle=20 * u.deg,
                    position=[x, y] * u.arcsec,
                    magnitude=mag * u.mag,
                    samples=SensingWfsMeasurements(
                        zero=noise * u.mas, shifted=shifted * u.mas, dtype=np.float64
                    ),
                    displacement=10 * u.mas,
                    sampling=sampling,
                    controller=controller,
                    photometry=photometry,
                    dtype=np.float64,
                )
            )
    sensor = build_sensing_response(
        points,
        field_radius=2 * u.arcsec,
        magnitude_intervals=[[8, 10]] * u.mag,
        dtype=np.float64,
    ).model
    sensing = SensingCalibration(
        response=sensor, blur=NoiseBlurModel(0.7, 0.3, 0.6, dtype=np.float64)
    )
    coma_model = ComaResponseModel(
        a0=0.08 * u.mas / u.nm,
        b=0.01 * u.mas / u.nm,
        m_half=9 * u.mag,
        magnitude_support=[8, 10] * u.mag,
        zenith_support=[0, 40] * u.deg,
        photometry=photometry,
        dtype=np.float64,
    )
    metrics = build_ngs_ho_metric_interpolator(
        NgsHoMetricSamples(
            x=xy[:, 0] * u.arcsec,
            y=xy[:, 1] * u.arcsec,
            zenith_angle=20 * u.deg,
            ee=np.full(9, 0.5) * u.one,
            fwhm=np.full(9, 100) * u.mas,
            sr=np.full(9, 0.2) * u.one,
        ),
        dtype=np.float64,
    )
    prior = ReconstructorNoiseModel(
        photons_per_frame=[1, 1e6] * u.photon,
        fwhm=[10, 200] * u.mas,
        variance=np.ones((2, 2)) * 0.1 * u.mas**2,
        dtype=np.float64,
    )
    base = measure_covariance_calibration(
        record,
        controller,
        settings=CovarianceCalibrationSettings(feedback=(0.2, 2.0), bootstrap_draws=3),
        dtype=np.float64,
    )
    bank = build_covariance_calibration([base], dtype=np.float64).model
    strengths = [0.2, 2.0]
    for indices in ([0], [0, 8], [0, 8, 2]):
        positions = record.positions[indices]
        weights = build_lo_reconstructor(
            bank,
            prior,
            ngs_positions=positions,
            zenith_angle=record.zenith_angle,
            photons_per_frame=np.full(len(indices), 1e4) * u.photon,
            ee=np.full(len(indices), 0.5) * u.one,
            fwhm=np.full(len(indices), 100) * u.mas,
            dtype=np.float64,
        )
        gains = sensor.predict(
            positions=positions,
            magnitudes=np.full(len(indices), 9) * u.mag,
            zenith_angle=record.zenith_angle,
        ).gain.value
        feedback = sum(g * weights[:, 2 * i : 2 * i + 2] for i, g in enumerate(gains))
        strengths.extend(_scalar_decomposition(feedback)["strengths"])
    measured = measure_covariance_calibration(
        record,
        controller,
        settings=CovarianceCalibrationSettings(
            feedback=tuple(sorted(set(strengths))), bootstrap_draws=3
        ),
        dtype=np.float64,
    )
    assert measured.discard == base.discard
    bank = build_covariance_calibration([measured], dtype=np.float64).model
    return E2ECalibratedModel(
        covariance=bank,
        coma=coma_model,
        sensing=sensing,
        reconstructor=prior,
        ngs_metrics=metrics,
        dtype=np.float64,
    ), record


@pytest.mark.parametrize("indices", [[0], [0, 8], [0, 8, 2]])
def test_complete_covariances_match_independent_matrix_replay(synthetic, indices):
    model, record = synthetic
    prediction = model.predict(
        ngs_positions=record.positions[indices],
        magnitudes=np.full(len(indices), 9) * u.mag,
        science_positions=record.positions,
        zenith_angle=record.zenith_angle,
    )
    weights = prediction.reconstructor
    atm_input = sum(
        record.tilt.value[:, :, idx]
        @ weights[:, 2 * g : 2 * g + 2].T
        * prediction.gains.value[g]
        for g, idx in enumerate(indices)
    )
    ho_input = sum(
        record.coma.value[:, :, idx]
        @ weights[:, 2 * g : 2 * g + 2].T
        * prediction.coma_response.value[g]
        for g, idx in enumerate(indices)
    )
    controller = model.covariance.controller
    atm = matrix_replay(
        atm_input, prediction.feedback, controller.gain, controller.delay
    )[:, model.covariance.discard :]
    ho = matrix_replay(
        ho_input, prediction.feedback, controller.gain, controller.delay
    )[:, model.covariance.discard :]
    science = record.tilt.value[:, model.covariance.discard :]
    for name, left, right in (
        ("atmospheric_command", atm, atm),
        ("ho_command", ho, ho),
        ("atmospheric_ho_command", atm, ho),
    ):
        np.testing.assert_allclose(
            getattr(prediction, name).value, pooled(left, right), rtol=1e-10, atol=1e-10
        )
    for j in range(len(record.positions)):
        for name, right in (
            ("science_atmospheric_command", atm),
            ("science_ho_command", ho),
        ):
            np.testing.assert_allclose(
                getattr(prediction, name).value[j],
                pooled(science[:, :, j], right),
                rtol=1e-10,
                atol=1e-10,
            )
        residual = science[:, :, j] - atm - ho
        np.testing.assert_allclose(
            prediction.noiseless.value[j],
            pooled(residual, residual),
            rtol=1e-10,
            atol=1e-10,
        )
    if len(indices) == 1:
        np.testing.assert_array_equal(weights, np.eye(2))
    else:
        assert np.ptp(prediction.gains.value) > 0.05
        # This fixture must exercise matrix ordering, not commuting scalar weights.
        assert any(
            np.linalg.norm(p @ weights[:, :2] - weights[:, :2] @ p) > 0.01
            for p in prediction.projectors
        )
    np.testing.assert_allclose(
        np.einsum(
            "c,scab->sab",
            prediction.mixture_weights,
            prediction.mixture_covariances.value,
        ),
        prediction.total.value,
        rtol=1e-13,
        atol=1e-13,
    )
    np.testing.assert_allclose(
        prediction.noise,
        model.sensing.noise_covariance(
            positions=record.positions[indices],
            magnitudes=np.full(len(indices), 9) * u.mag,
            zenith_angle=record.zenith_angle,
            reconstructor=weights,
        ),
        rtol=1e-13,
    )
    assert np.linalg.norm(atm.mean((0, 1))) > 0.1


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("indices", [[0], [0, 8], [0, 8, 2]])
def test_precision_immutability_and_input_rejection(synthetic, dtype, indices):
    model, record = synthetic
    model = model.prepare(dtype=dtype)
    args = {
        "ngs_positions": record.positions[indices],
        "magnitudes": np.full(len(indices), 9) * u.mag,
        "science_positions": record.positions,
        "zenith_angle": record.zenith_angle,
    }
    result = model.predict(**args)
    for value in (
        result.total,
        result.noise,
        result.mixture_covariances,
        result.gains,
        result.reconstructor,
        result.projectors,
        result.photons_per_frame,
        result.metrics.ee,
        result.metrics.fwhm,
        result.metrics.sr,
    ):
        assert value.dtype == dtype
        with pytest.raises(ValueError):
            (
                value.value if isinstance(value, u.Quantity) else value
            ).flags.writeable = True
    for name in ("total", "noise", "mixture_covariances", "gains", "photons_per_frame"):
        expected = getattr(result, name).copy()
        backing = getattr(result, name)
        while isinstance(backing, np.ndarray):
            if isinstance(backing, u.Quantity):
                backing *= u.s
                assert getattr(result, name).unit == expected.unit
            backing = backing.base
        np.testing.assert_array_equal(getattr(result, name), expected)
    with pytest.raises(ValueError, match="support|outside"):
        model.predict(**(args | {"magnitudes": np.full(len(indices), 12) * u.mag}))
    with pytest.raises(ValueError, match="distinct"):
        model.predict(
            **(
                args
                | {
                    "ngs_positions": record.positions[[0, 0]],
                    "magnitudes": [9, 9] * u.mag,
                }
            )
        )


def test_feedback_limits_and_scalar_closed_form():
    for b in (0.5, 1.0, 1.7):
        values = _scalar_decomposition(np.eye(2) * b)
        np.testing.assert_allclose(values["strengths"], b)
        np.testing.assert_allclose(values["projectors"].sum(0), np.eye(2))
        assert not values["symmetric_feedback_fallback"]
    feedback = np.array([[0.8, 0.2], [-0.2, 0.8]])
    values = _scalar_decomposition(feedback)
    assert values["symmetric_feedback_fallback"]
    np.testing.assert_allclose(values["effective_feedback"], np.eye(2) * 0.8)
    assert values["feedback_approximation_relative_error"] > 0.1
    with pytest.raises(ValueError, match="defective"):
        _scalar_decomposition(np.array([[0.8, 1.0], [0, 0.8]]))


@pytest.mark.parametrize(
    "dtype,feedback",
    [
        (
            np.float32,
            [
                [-62.42650604248047, 7.819972991943359],
                [-514.409912109375, 64.42650604248047],
            ],
        ),
        (
            np.float64,
            [
                [509028.38564056694, 806947.8356399821],
                [-321097.4339206497, -509026.38564056694],
            ],
        ),
    ],
)
def test_nonorthogonal_projectors_allow_matrix_product_roundoff(dtype, feedback):
    feedback = np.array(feedback, dtype=dtype)
    controller = FixedGainIntegrator(
        gain=0.3, delay=2, frame_rate=500 * u.Hz, dtype=dtype
    )
    assert np.max(np.abs(np.linalg.eigvals(controller.transition(feedback)))) < 1
    decomposition = _scalar_decomposition(feedback)
    projectors = decomposition["projectors"]
    tolerance = max(1e-10, 32 * np.finfo(dtype).eps)
    assert not decomposition["symmetric_feedback_fallback"]
    assert decomposition["eigenbasis_condition"] < min(
        1e10, 1 / (100 * np.finfo(dtype).eps)
    )
    np.testing.assert_allclose(
        projectors.sum(0), np.eye(2), rtol=tolerance, atol=tolerance
    )
    np.testing.assert_allclose(
        np.einsum("p,pab->ab", decomposition["strengths"], projectors),
        feedback,
        rtol=tolerance,
        atol=tolerance,
    )
    # Multiplying large non-orthogonal projectors needs a product-roundoff allowance.
    assert not np.allclose(
        projectors @ projectors, projectors, rtol=tolerance, atol=tolerance
    )


def test_configured_native_free_loading_and_conflicts(synthetic, tmp_path, monkeypatch):
    model, record = synthetic
    families = ("covariance", "coma", "sensing", "reconstructor")
    for family, save in zip(
        families,
        (
            save_covariance_calibration,
            save_coma_calibration,
            save_sensing_calibration,
            save_reconstructor_calibration,
        ),
        strict=True,
    ):
        save(getattr(model, family), tmp_path / family / "model.json")
    save_ngs_ho_metric_interpolator(model.ngs_metrics, tmp_path / "metrics.pkl")
    ini = tmp_path / "model.ini"
    ini.write_text(
        "[e2e]\nprecision = float64\n"
        + "".join(f"{f}_model = {f}/model.json\n" for f in families)
        + "ngs_metrics = metrics.pkl\nscience_psf = missing-science.pkl\nngs_psf = missing-spots.pkl\n[native]\nsource = unavailable\n"
    )
    original_import = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name.startswith(("girmos_oomao", "ao_tools", "oomao", "mastsel")):
            raise AssertionError(f"native/downstream dependency: {name}")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    loaded = load_e2e_calibrated_model(ini)
    assert len(loaded.product_identities) == 5
    args = {
        "ngs_positions": record.positions[[0, 8]],
        "magnitudes": [9, 9] * u.mag,
        "science_positions": record.positions,
        "zenith_angle": 20 * u.deg,
    }
    np.testing.assert_array_equal(
        loaded.predict(**args).total, model.predict(**args).total
    )
    with pytest.raises(TypeError):
        loaded.product_identities["other"] = None
    np.testing.assert_allclose(
        loaded.predict(
            **(
                args
                | {
                    "ngs_positions": args["ngs_positions"].to(u.rad),
                    "science_positions": args["science_positions"].to(u.arcmin),
                    "zenith_angle": args["zenith_angle"].to(u.rad),
                }
            )
        ).total,
        loaded.predict(**args).total,
        rtol=1e-12,
        atol=1e-12,
    )
    code = """
import builtins
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.startswith(('mastsel','girmos_oomao','ao_tools','oomao')):
        raise AssertionError(name)
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
from astropy import units as u
from hybrid_ao_psf import load_e2e_calibrated_model
import sys
model = load_e2e_calibrated_model(sys.argv[1])
result = model.predict(ngs_positions=[[-1,-1],[1,1]]*u.arcsec,
    magnitudes=[9,9]*u.mag, science_positions=[[0,0]]*u.arcsec, zenith_angle=20*u.deg)
assert result.total.shape == (1,2,2)
"""
    subprocess.run(
        [sys.executable, "-c", code, str(ini)],
        check=True,
        capture_output=True,
        text=True,
    )
    sensing_path = tmp_path / "sensing/model.json"
    header = json.loads(sensing_path.read_text())
    header["controller"]["gain"] = 0.2
    sensing_path.write_text(json.dumps(header))
    with pytest.raises(ValueError, match="controller"):
        replace(model, sensing=load_sensing_calibration(sensing_path))
    with pytest.raises(ValueError, match="photometry"):
        replace(
            model,
            coma=replace(
                model.coma,
                photometry=replace(model.coma.photometry, magnitude_band="other"),
            ),
        )
    (tmp_path / "coma/model.json").unlink()
    with pytest.raises(FileNotFoundError):
        load_e2e_calibrated_model(ini)


@pytest.mark.parametrize("indices", [[0], [0, 8], [0, 8, 2]])
def test_noise_covariance_matches_independent_impulse_sum(synthetic, indices):
    model, record = synthetic
    result = model.predict(
        ngs_positions=record.positions[indices],
        magnitudes=np.full(len(indices), 9) * u.mag,
        science_positions=record.positions[[4]],
        zenith_angle=20 * u.deg,
    )
    gain, delay = model.covariance.controller.gain, model.covariance.controller.delay
    impulse = np.zeros((2048, 2, 2))
    for k in range(1, len(impulse)):
        impulse[k] = impulse[k - 1]
        if k >= delay:
            impulse[k] -= gain * result.feedback @ impulse[k - delay]
        if k == delay:
            impulse[k] += gain * np.eye(2)
    innovation = sum(
        v
        * result.reconstructor[:, 2 * g : 2 * g + 2]
        @ result.reconstructor[:, 2 * g : 2 * g + 2].T
        for g, v in enumerate(result.noise_variances.value)
    )
    expected = sum(a @ innovation @ a.T for a in impulse)
    np.testing.assert_allclose(result.noise.value, expected, rtol=1e-12, atol=1e-12)


def test_unsupported_spatial_and_zenith_queries_are_rejected(synthetic):
    model, record = synthetic
    args = {
        "ngs_positions": record.positions[[0]],
        "magnitudes": [9] * u.mag,
        "science_positions": record.positions[[4]],
        "zenith_angle": 20 * u.deg,
    }
    with pytest.raises(ValueError, match="support|outside"):
        model.predict(**(args | {"science_positions": [[10, 10]] * u.arcsec}))
    with pytest.raises(ValueError, match="support|fixed"):
        model.predict(**(args | {"zenith_angle": 60 * u.deg}))


def test_complex_feedback_bank_uses_symmetric_part_but_noise_uses_actual(
    synthetic, monkeypatch
):
    import hybrid_ao_psf._e2e_model as implementation

    model, record = synthetic
    indices = [0, 8]
    gains = model.sensing.predict(
        positions=record.positions[indices],
        magnitudes=[9, 9] * u.mag,
        zenith_angle=20 * u.deg,
    ).gain.value
    b = float(model.covariance.feedback[1])
    actual = np.array([[b, 0.15], [-0.15, b]])
    first = np.array([[0.3, 0.1], [-0.07, 0.4]])
    second = (actual - gains[0] * first) / gains[1]
    weights = np.hstack((first, second))
    monkeypatch.setattr(
        implementation, "build_lo_reconstructor", lambda *args, **kwargs: weights
    )
    result = model.predict(
        ngs_positions=record.positions[indices],
        magnitudes=[9, 9] * u.mag,
        science_positions=record.positions,
        zenith_angle=20 * u.deg,
    )
    assert result.symmetric_feedback_fallback
    np.testing.assert_allclose(result.effective_feedback, b * np.eye(2), atol=1e-15)
    atm = sum(
        record.tilt.value[:, :, idx] @ weights[:, 2 * g : 2 * g + 2].T * gains[g]
        for g, idx in enumerate(indices)
    )
    ho = sum(
        record.coma.value[:, :, idx]
        @ weights[:, 2 * g : 2 * g + 2].T
        * result.coma_response.value[g]
        for g, idx in enumerate(indices)
    )
    effective = matrix_replay(atm + ho, result.effective_feedback, 0.3, 2)[
        :, model.covariance.discard :
    ]
    for j in range(len(record.positions)):
        residual = record.tilt.value[:, model.covariance.discard :, j] - effective
        np.testing.assert_allclose(
            result.noiseless.value[j],
            pooled(residual, residual),
            rtol=1e-11,
            atol=1e-11,
        )
    innovation = sum(
        v * weights[:, 2 * g : 2 * g + 2] @ weights[:, 2 * g : 2 * g + 2].T
        for g, v in enumerate(result.noise_variances.value)
    )
    impulse = np.zeros((2048, 2, 2))
    for k in range(1, len(impulse)):
        impulse[k] = impulse[k - 1]
        if k >= 2:
            impulse[k] -= 0.3 * actual @ impulse[k - 2]
        if k == 2:
            impulse[k] += 0.3 * np.eye(2)
    expected = sum(a @ innovation @ a.T for a in impulse)
    np.testing.assert_allclose(result.noise.value, expected, rtol=1e-12, atol=1e-12)
    approximate = model.sensing.controller.noise_covariance(
        result.effective_feedback, innovation * u.mas**2
    )
    assert np.linalg.norm(expected - approximate.value) > 0.001


@pytest.mark.parametrize(
    "feedback,error",
    [
        (np.eye(2) * 2.5, "support"),
        (np.eye(2) * 4, "Unstable"),
        (np.array([[0.8, 1.0], [0, 0.8]]), "defective"),
    ],
)
def test_model_rejects_unsupported_feedback(synthetic, monkeypatch, feedback, error):
    import hybrid_ao_psf._e2e_model as implementation

    model, record = synthetic
    g = model.sensing.predict(
        positions=record.positions[[0]], magnitudes=[9] * u.mag, zenith_angle=20 * u.deg
    ).gain.value[0]
    monkeypatch.setattr(
        implementation, "build_lo_reconstructor", lambda *args, **kwargs: feedback / g
    )
    with pytest.raises(ValueError, match=error):
        model.predict(
            ngs_positions=record.positions[[0]],
            magnitudes=[9] * u.mag,
            science_positions=record.positions[[4]],
            zenith_angle=20 * u.deg,
        )


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_direct_model_construction_prepares_nested_provider_precision(synthetic, dtype):
    model, _ = synthetic
    # Outer dtype agrees while the public nested blur/photometry disagree.
    other = np.float64 if dtype == np.float32 else np.float32
    response = model.sensing.response.prepare(dtype=dtype)
    sensing = SensingCalibration(
        response=response, blur=model.sensing.blur.prepare(dtype=other)
    )
    coma = replace(
        model.coma.prepare(dtype=dtype),
        photometry=replace(model.coma.photometry, dtype=other),
    )
    prepared = replace(model, sensing=sensing, coma=coma, dtype=dtype)
    assert prepared.sensing.response.dtype == dtype
    assert prepared.sensing.blur.dtype == dtype
    assert prepared.sensing.controller.dtype == dtype
    assert prepared.sensing.response.photometry.dtype == dtype
    assert prepared.coma.photometry.dtype == dtype
    assert prepared.prepare(dtype=dtype) is prepared
    # Preparation owns fresh state and does not change supplied providers.
    assert sensing.blur.dtype == other
    assert coma.photometry.dtype == other


def test_direct_model_construction_validates_nested_values_after_rounding(synthetic):
    model, _ = synthetic
    sensing = SensingCalibration(
        response=model.sensing.response.prepare(dtype=np.float32),
        blur=NoiseBlurModel(0.999999999, 0.3, 0.6, dtype=np.float64),
    )
    with pytest.raises(ValueError, match="weight"):
        replace(model, sensing=sensing, dtype=np.float32)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_prepared_controller_scalar_storage_and_units_are_immutable(synthetic, dtype):
    model, _ = synthetic
    controller = model.prepare(dtype=dtype).covariance.controller
    backing = controller.frame_rate
    assert backing.shape == ()
    assert backing.dtype == dtype
    while isinstance(backing, np.ndarray):
        with pytest.raises(ValueError):
            backing.flags.writeable = True
        if isinstance(backing, u.Quantity):
            backing *= u.s
            assert controller.frame_rate.unit == u.Hz
        backing = backing.base
    assert controller.frame_rate == 500 * u.Hz
