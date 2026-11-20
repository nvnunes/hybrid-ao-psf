"""Independent matrix simulation and configured calibrated-model contracts."""

import builtins
import json
import subprocess
import sys
from dataclasses import replace

import numpy as np
import pytest
from ao_core.models.control import TipTiltControl
from astropy import units as u

from hybrid_ao_psf import (
    CalibrationPhotometry,
    ComaResponseModel,
    CovarianceCalibrationSettings,
    CovarianceFunctions,
    CovarianceTelemetry,
    E2ECalibratedModel,
    FixedGainIntegrator,
    LowOrderDistribution,
    NoiseBlurModel,
    PsfMetricSamples,
    ReconstructorNoiseModel,
    SensingCalibrationMeasurements,
    SensingModel,
    SensingWfsMeasurements,
    SensingWfsSampling,
    build_covariance_calibration,
    build_lo_reconstructor,
    build_psf_metric_interpolator,
    build_sensing_response,
    load_e2e_calibrated_model,
    load_sensing_calibration,
    measure_covariance_calibration,
    save_coma_calibration,
    save_covariance_calibration,
    save_psf_metric_interpolator,
    save_reconstructor_calibration,
    save_sensing_calibration,
)
from hybrid_ao_psf._lo_model import _ReconstructorCovariances, _solve_control


def scalar_feedback(feedback):
    controller = FixedGainIntegrator(
        gain=0.3, delay=2, frame_rate=500 * u.Hz, dtype=feedback.dtype
    )
    return TipTiltControl(
        reconstructor=feedback * u.one,
        sensing_gains=[1.0] * u.one,
        controller=controller,
    ).scalar_feedback


def pooled(left, right):
    left, right = left.reshape(-1, 2), right.reshape(-1, 2)
    return (left - left.mean(0)).T @ (right - right.mean(0)) / len(left)


def simulate_matrix_loop(inputs, feedback, gain, delay):
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
    sensing = SensingModel(
        response=sensor, blur=NoiseBlurModel(0.7, 0.3, 0.6, dtype=np.float64)
    )
    coma_model = ComaResponseModel(
        a0=0.08 * u.mas / u.nm,
        b=0.01 * u.mas / u.nm,
        m_half=9 * u.mag,
        magnitude_range=[8, 10] * u.mag,
        zenith_range=[0, 40] * u.deg,
        photometry=photometry,
        dtype=np.float64,
    )
    metrics = build_psf_metric_interpolator(
        PsfMetricSamples(
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
        settings=CovarianceCalibrationSettings(feedback=(0.2, 2.0)),
        dtype=np.float64,
    )
    covariances = build_covariance_calibration([base], dtype=np.float64).model
    strengths = [0.2, 2.0]
    for indices in ([0], [0, 8], [0, 8, 2]):
        positions = record.positions[indices]
        weights = build_lo_reconstructor(
            covariances,
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
        strengths.extend(scalar_feedback(feedback).strengths)
    measured = measure_covariance_calibration(
        record,
        controller,
        settings=CovarianceCalibrationSettings(feedback=tuple(sorted(set(strengths)))),
        dtype=np.float64,
    )
    assert measured.discard == base.discard
    covariances = build_covariance_calibration([measured], dtype=np.float64).model
    return E2ECalibratedModel(
        covariance=covariances,
        coma=coma_model,
        sensing=sensing,
        reconstructor=prior,
        ngs_metrics=metrics,
        dtype=np.float64,
    ), record


@pytest.mark.parametrize("indices,expected_tilt_queries", [([0], 1), ([0, 8], 3)])
def test_reconstructor_uses_one_evaluated_prior(
    synthetic, monkeypatch, indices, expected_tilt_queries
):
    model, record = synthetic
    priors = []
    original_prior = ReconstructorNoiseModel.build_measurement_noise_covariance
    original_tilt = CovarianceFunctions.C_t_t
    tilt_queries = []

    def build_measurement_noise_covariance(self, **kwargs):
        prior = original_prior(self, **kwargs)
        priors.append(prior)
        return prior

    def tilt_covariance(self, *args, **kwargs):
        tilt_queries.append(args)
        return original_tilt(self, *args, **kwargs)

    monkeypatch.setattr(
        ReconstructorNoiseModel,
        "build_measurement_noise_covariance",
        build_measurement_noise_covariance,
    )
    monkeypatch.setattr(CovarianceFunctions, "C_t_t", tilt_covariance)
    prediction = model.predict(
        diagnostics_level="debug",
        ngs_positions=record.positions[indices],
        ngs_magnitude=np.full(len(indices), 9) * u.mag,
        science_positions=record.positions,
        zenith_angle=record.zenith_angle,
    )

    assert len(priors) == 1
    np.testing.assert_array_equal(
        prediction.diagnostics.ngs.reconstructor_noise_covariance, priors[0]
    )
    assert len(tilt_queries) == expected_tilt_queries
    if len(indices) == 1:
        np.testing.assert_array_equal(
            prediction.diagnostics.control.reconstructor, np.eye(2)
        )


def test_control_solve_rejects_reconstruction_for_wrong_ngs_count(synthetic):
    model, _ = synthetic
    one_ngs_covariance = np.eye(2) * u.mas**2
    with pytest.raises(ValueError, match="two axes per NGS"):
        _solve_control(
            reconstruction_covariances=_ReconstructorCovariances(
                ngs_covariance=one_ngs_covariance,
                centre_ngs_covariance=one_ngs_covariance,
                noise_prior=one_ngs_covariance,
            ),
            sensing_gains=[0.4, 0.6] * u.one,
            controller=model.covariance.controller,
            dtype=model.dtype,
        )


@pytest.mark.parametrize("indices", [[0], [0, 8], [0, 8, 2]])
def test_complete_covariances_match_independent_loop_simulation(synthetic, indices):
    model, record = synthetic
    prediction = model.predict(
        diagnostics_level="debug",
        ngs_positions=record.positions[indices],
        ngs_magnitude=np.full(len(indices), 9) * u.mag,
        science_positions=record.positions,
        zenith_angle=record.zenith_angle,
    )
    diagnostics = prediction.diagnostics
    weights = diagnostics.control.reconstructor
    atm_input = sum(
        record.tilt.value[:, :, idx]
        @ weights[:, 2 * g : 2 * g + 2].T
        * diagnostics.ngs.sensing.gain.value[g]
        for g, idx in enumerate(indices)
    )
    ho_input = sum(
        record.coma.value[:, :, idx]
        @ weights[:, 2 * g : 2 * g + 2].T
        * diagnostics.ngs.coma_response_coefficients.value[g]
        for g, idx in enumerate(indices)
    )
    controller = model.covariance.controller
    atm = simulate_matrix_loop(
        atm_input, diagnostics.control.feedback, controller.gain, controller.delay
    )[:, model.covariance.discard :]
    ho = simulate_matrix_loop(
        ho_input, diagnostics.control.feedback, controller.gain, controller.delay
    )[:, model.covariance.discard :]
    science = record.tilt.value[:, model.covariance.discard :]
    terms = diagnostics.covariances.terms
    for actual, left, right in (
        (terms.atmospheric_command, atm, atm),
        (terms.ho_command, ho, ho),
        (terms.atmospheric_ho_command, atm, ho),
    ):
        np.testing.assert_allclose(
            actual.value, pooled(left, right), rtol=1e-10, atol=1e-10
        )
    for j in range(len(record.positions)):
        for actual, right in (
            (terms.science_atmospheric_command, atm),
            (terms.science_ho_command, ho),
        ):
            np.testing.assert_allclose(
                actual.value[j],
                pooled(science[:, :, j], right),
                rtol=1e-10,
                atol=1e-10,
            )
        science_tilt = science[:, :, j]
        np.testing.assert_allclose(
            diagnostics.covariances.terms.science_tilt.value[j],
            pooled(science_tilt, science_tilt),
            rtol=1e-10,
            atol=1e-10,
        )
        atmospheric_residual = science_tilt - atm
        atmospheric_covariance = pooled(atmospheric_residual, atmospheric_residual)
        residual = atmospheric_residual - ho
        noiseless_covariance = pooled(residual, residual)
        np.testing.assert_allclose(
            diagnostics.covariances.atmospheric_residual.value[j],
            atmospheric_covariance,
            rtol=1e-10,
            atol=1e-10,
        )
        np.testing.assert_allclose(
            diagnostics.covariances.ho_correction.value[j],
            noiseless_covariance - atmospheric_covariance,
            rtol=1e-10,
            atol=1e-10,
        )
        np.testing.assert_allclose(
            diagnostics.covariances.noiseless.value[j],
            noiseless_covariance,
            rtol=1e-10,
            atol=1e-10,
        )
    if len(indices) == 1:
        np.testing.assert_array_equal(weights, np.eye(2))
    else:
        assert np.ptp(diagnostics.ngs.sensing.gain.value) > 0.05
        # This fixture must exercise matrix ordering, not commuting scalar weights.
        assert any(
            np.linalg.norm(p @ weights[:, :2] - weights[:, :2] @ p) > 0.01
            for p in diagnostics.control.scalar_feedback.projectors
        )
    assert isinstance(prediction.displacement_distribution, LowOrderDistribution)
    np.testing.assert_allclose(
        prediction.displacement_distribution.covariance.value,
        (
            diagnostics.covariances.noiseless + diagnostics.covariances.noise_command
        ).value,
        rtol=1e-13,
        atol=1e-13,
    )
    assert np.linalg.norm(atm.mean((0, 1))) > 0.1


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("indices", [[0], [0, 8], [0, 8, 2]])
def test_precision_immutability_and_input_rejection(synthetic, dtype, indices):
    model, record = synthetic
    model = model.prepare(dtype=dtype)
    args = {
        "ngs_positions": record.positions[indices],
        "ngs_magnitude": np.full(len(indices), 9) * u.mag,
        "science_positions": record.positions,
        "zenith_angle": record.zenith_angle,
    }
    result = model.predict(**args, diagnostics_level="debug")
    diagnostics = result.diagnostics
    with pytest.raises(TypeError, match="unexpected keyword argument.*magnitudes"):
        model.predict(**(args | {"magnitudes": args["ngs_magnitude"]}))
    from dataclasses import fields

    assert [item.name for item in fields(result)] == [
        "displacement_distribution",
        "diagnostics",
    ]
    assert result.displacement_distribution.covariance.dtype == dtype
    for value in (
        diagnostics.covariances.terms.science_tilt,
        diagnostics.covariances.noiseless,
        diagnostics.covariances.atmospheric_residual,
        diagnostics.covariances.ho_correction,
        diagnostics.covariances.noise_command,
        result.displacement_distribution.weights,
        result.displacement_distribution.covariances,
        diagnostics.ngs.sensing.gain,
        diagnostics.ngs.sensing.variance,
        diagnostics.control.reconstructor,
        diagnostics.control.scalar_feedback.projectors,
        diagnostics.ngs.photons_per_frame,
        diagnostics.ngs.metrics.ee,
        diagnostics.ngs.metrics.fwhm,
        diagnostics.ngs.metrics.sr,
    ):
        assert value.dtype == dtype
        with pytest.raises(ValueError):
            (
                value.value if isinstance(value, u.Quantity) else value
            ).flags.writeable = True
    for owner, name in (
        (diagnostics.covariances.terms, "science_tilt"),
        (diagnostics.covariances, "noiseless"),
        (diagnostics.covariances, "atmospheric_residual"),
        (diagnostics.covariances, "ho_correction"),
        (diagnostics.covariances, "noise_command"),
        (result.displacement_distribution, "covariances"),
        (diagnostics.ngs.sensing, "gain"),
        (diagnostics.ngs.sensing, "variance"),
        (diagnostics.ngs, "photons_per_frame"),
    ):
        expected = getattr(owner, name).copy()
        backing = getattr(owner, name)
        while isinstance(backing, np.ndarray):
            if isinstance(backing, u.Quantity):
                backing *= u.s
                assert getattr(owner, name).unit == expected.unit
            backing = backing.base
        np.testing.assert_array_equal(getattr(owner, name), expected)
    with pytest.raises(ValueError, match="range|outside"):
        model.predict(**(args | {"ngs_magnitude": np.full(len(indices), 12) * u.mag}))
    with pytest.raises(ValueError, match="distinct"):
        model.predict(
            **(
                args
                | {
                    "ngs_positions": record.positions[[0, 0]],
                    "ngs_magnitude": [9, 9] * u.mag,
                }
            )
        )


def test_feedback_limits_and_scalar_closed_form():
    for b in (0.5, 1.0, 1.7):
        values = scalar_feedback(np.eye(2) * b)
        np.testing.assert_allclose(values.strengths, b)
        np.testing.assert_allclose(values.projectors.sum(0), np.eye(2))
        assert not values.symmetric_fallback
    feedback = np.array([[0.8, 0.2], [-0.2, 0.8]])
    values = scalar_feedback(feedback)
    assert values.symmetric_fallback
    np.testing.assert_allclose(values.effective_feedback, np.eye(2) * 0.8)
    assert values.approximation_relative_error > 0.1
    with pytest.raises(ValueError, match="defective"):
        scalar_feedback(np.array([[0.8, 1.0], [0, 0.8]]))


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
    decomposition = scalar_feedback(feedback)
    projectors = decomposition.projectors
    tolerance = max(1e-10, 32 * np.finfo(dtype).eps)
    assert not decomposition.symmetric_fallback
    assert decomposition.eigenbasis_condition < min(
        1e10, 1 / (100 * np.finfo(dtype).eps)
    )
    np.testing.assert_allclose(
        projectors.sum(0), np.eye(2), rtol=tolerance, atol=tolerance
    )
    np.testing.assert_allclose(
        np.einsum("p,pab->ab", decomposition.strengths, projectors),
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
    save_psf_metric_interpolator(model.ngs_metrics, tmp_path / "metrics.pkl")
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
    assert not hasattr(loaded, "model_files")
    args = {
        "ngs_positions": record.positions[[0, 8]],
        "ngs_magnitude": [9, 9] * u.mag,
        "science_positions": record.positions,
        "zenith_angle": 20 * u.deg,
    }
    np.testing.assert_array_equal(
        loaded.predict(**args).displacement_distribution.covariance,
        model.predict(**args).displacement_distribution.covariance,
    )
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
        ).displacement_distribution.covariance,
        loaded.predict(**args).displacement_distribution.covariance,
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
    ngs_magnitude=[9,9]*u.mag, science_positions=[[0,0]]*u.arcsec, zenith_angle=20*u.deg)
assert result.displacement_distribution.covariance.shape == (1,2,2)
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
        diagnostics_level="debug",
        ngs_positions=record.positions[indices],
        ngs_magnitude=np.full(len(indices), 9) * u.mag,
        science_positions=record.positions[[4]],
        zenith_angle=20 * u.deg,
    )
    diagnostics = result.diagnostics
    gain, delay = model.covariance.controller.gain, model.covariance.controller.delay
    impulse = np.zeros((2048, 2, 2))
    for k in range(1, len(impulse)):
        impulse[k] = impulse[k - 1]
        if k >= delay:
            impulse[k] -= gain * diagnostics.control.feedback @ impulse[k - delay]
        if k == delay:
            impulse[k] += gain * np.eye(2)
    innovation = sum(
        v
        * diagnostics.control.reconstructor[:, 2 * g : 2 * g + 2]
        @ diagnostics.control.reconstructor[:, 2 * g : 2 * g + 2].T
        for g, v in enumerate(diagnostics.ngs.sensing.variance.value)
    )
    expected = sum(a @ innovation @ a.T for a in impulse)
    np.testing.assert_allclose(
        diagnostics.covariances.noise_command.value, expected, rtol=1e-12, atol=1e-12
    )


def test_prediction_only_builds_debug_diagnostics_when_requested(
    synthetic, monkeypatch
):
    model, record = synthetic
    query = {
        "ngs_positions": record.positions[[0, 8]],
        "ngs_magnitude": [9, 9] * u.mag,
        "science_positions": record.positions,
        "zenith_angle": record.zenith_angle,
    }
    debug = model.predict(**query, diagnostics_level="debug")
    assert debug.diagnostics is not None

    def unexpected_diagnostics(**kwargs):
        pytest.fail("Prediction built a debug record without a debug request.")

    monkeypatch.setattr(
        "hybrid_ao_psf._e2e_model.E2ELowOrderDiagnostics", unexpected_diagnostics
    )
    for prediction in (
        model.predict(**query),
        model.predict(**query, diagnostics_level="validation"),
    ):
        assert prediction.diagnostics is None
        np.testing.assert_array_equal(
            prediction.displacement_distribution.covariances,
            debug.displacement_distribution.covariances,
        )
        np.testing.assert_array_equal(
            prediction.displacement_distribution.weights,
            debug.displacement_distribution.weights,
        )


def test_unsupported_spatial_and_zenith_queries_are_rejected(synthetic):
    model, record = synthetic
    args = {
        "ngs_positions": record.positions[[0]],
        "ngs_magnitude": [9] * u.mag,
        "science_positions": record.positions[[4]],
        "zenith_angle": 20 * u.deg,
    }
    with pytest.raises(ValueError, match="range|outside"):
        model.predict(**(args | {"science_positions": [[10, 10]] * u.arcsec}))
    with pytest.raises(ValueError, match="range|fixed"):
        model.predict(**(args | {"zenith_angle": 60 * u.deg}))


def test_complex_feedback_interpolation_uses_symmetric_part_but_noise_uses_actual(
    synthetic, monkeypatch
):
    import hybrid_ao_psf._lo_model as implementation

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
        implementation,
        "lo_reconstructor",
        lambda *args, **kwargs: weights,
    )
    result = model.predict(
        diagnostics_level="debug",
        ngs_positions=record.positions[indices],
        ngs_magnitude=[9, 9] * u.mag,
        science_positions=record.positions,
        zenith_angle=20 * u.deg,
    )
    diagnostics = result.diagnostics
    assert diagnostics.control.scalar_feedback.symmetric_fallback
    np.testing.assert_allclose(
        diagnostics.control.scalar_feedback.effective_feedback,
        b * np.eye(2),
        atol=1e-15,
    )
    atm = sum(
        record.tilt.value[:, :, idx] @ weights[:, 2 * g : 2 * g + 2].T * gains[g]
        for g, idx in enumerate(indices)
    )
    ho = sum(
        record.coma.value[:, :, idx]
        @ weights[:, 2 * g : 2 * g + 2].T
        * diagnostics.ngs.coma_response_coefficients.value[g]
        for g, idx in enumerate(indices)
    )
    effective = simulate_matrix_loop(
        atm + ho, diagnostics.control.scalar_feedback.effective_feedback, 0.3, 2
    )[:, model.covariance.discard :]
    for j in range(len(record.positions)):
        residual = record.tilt.value[:, model.covariance.discard :, j] - effective
        np.testing.assert_allclose(
            diagnostics.covariances.noiseless.value[j],
            pooled(residual, residual),
            rtol=1e-11,
            atol=1e-11,
        )
    innovation = sum(
        v * weights[:, 2 * g : 2 * g + 2] @ weights[:, 2 * g : 2 * g + 2].T
        for g, v in enumerate(diagnostics.ngs.sensing.variance.value)
    )
    impulse = np.zeros((2048, 2, 2))
    for k in range(1, len(impulse)):
        impulse[k] = impulse[k - 1]
        if k >= 2:
            impulse[k] -= 0.3 * actual @ impulse[k - 2]
        if k == 2:
            impulse[k] += 0.3 * np.eye(2)
    expected = sum(a @ innovation @ a.T for a in impulse)
    np.testing.assert_allclose(
        diagnostics.covariances.noise_command.value, expected, rtol=1e-12, atol=1e-12
    )
    approximate = model.sensing.controller.noise_covariance(
        diagnostics.control.scalar_feedback.effective_feedback, innovation * u.mas**2
    )
    assert np.linalg.norm(expected - approximate.value) > 0.001


@pytest.mark.parametrize(
    "feedback,error",
    [
        (np.eye(2) * 2.5, "range"),
        (np.eye(2) * 4, "Unstable"),
        (np.array([[0.7, -0.6], [0.6, -0.1]]), "Unstable"),  # Symmetric part only.
        (np.array([[0.8, 1.0], [0, 0.8]]), "defective"),
    ],
)
def test_model_rejects_unsupported_feedback(synthetic, monkeypatch, feedback, error):
    import hybrid_ao_psf._lo_model as implementation

    model, record = synthetic
    positions = record.positions[[0, 8]]
    magnitudes = [9, 9] * u.mag
    gain = model.sensing.predict(
        positions=positions, magnitudes=magnitudes, zenith_angle=20 * u.deg
    ).gain.value[0]
    weights = np.hstack((feedback / gain, np.zeros_like(feedback)))
    monkeypatch.setattr(
        implementation,
        "lo_reconstructor",
        lambda *args, **kwargs: weights,
    )
    with pytest.raises(ValueError, match=error):
        model.predict(
            ngs_positions=positions,
            ngs_magnitude=magnitudes,
            science_positions=record.positions[[4]],
            zenith_angle=20 * u.deg,
        )


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_direct_model_construction_prepares_nested_provider_precision(synthetic, dtype):
    model, _ = synthetic
    # Outer dtype agrees while the public nested blur/photometry disagree.
    other = np.float64 if dtype == np.float32 else np.float32
    response = model.sensing.response.prepare(dtype=dtype)
    sensing = SensingModel(
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
    sensing = SensingModel(
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


@pytest.mark.parametrize("family", ["covariance", "coma", "sensing", "reconstructor"])
def test_calibrated_models_use_grid_precision_without_former_names(
    synthetic, tmp_path, family
):
    import json

    import hybrid_ao_psf as api

    model, _ = synthetic
    source = getattr(model, family)
    path = tmp_path / "model.json"
    getattr(api, f"save_{family}_calibration")(source, path)
    header = json.loads(path.read_text())
    assert header["grid_precision"] == "float64"
    assert "support_precision" not in header
    loaded = getattr(api, f"load_{family}_calibration")(path)
    response = loaded.response if family == "sensing" else loaded
    assert response.grid_precision == np.dtype("float64")
    assert not hasattr(response, "support_precision")
    if family == "coma":
        np.testing.assert_array_equal(loaded.magnitude_range, source.magnitude_range)
        np.testing.assert_array_equal(loaded.zenith_range, source.zenith_range)
        assert not hasattr(loaded, "magnitude_support")
        assert not hasattr(loaded, "zenith_support")
    if family == "sensing":
        assert not hasattr(response, "support_mask")
        with np.load(path.parent / header["array_file"]) as arrays:
            assert "measured_mask" in arrays.files
            assert "support" not in arrays.files
            np.testing.assert_array_equal(
                arrays["measured_mask"], source.response.measured_mask
            )
    header["support_precision"] = header.pop("grid_precision")
    path.write_text(json.dumps(header))
    with pytest.raises(ValueError):
        getattr(api, f"load_{family}_calibration")(path)


def test_sensing_measured_mask_rejects_interpolation_through_unmeasured_points(
    synthetic,
):
    model, _ = synthetic
    source = model.sensing.response
    with pytest.raises(ValueError):
        source.measured_mask.flags.writeable = True
    arrays = {name: values.copy() for name, values in source._arrays.items()}
    arrays["measured_mask"][0, 0, 0, 0] = False
    masked = type(source)._from_arrays(
        arrays,
        source.controller,
        source.photometry,
        source.field_radius,
        source.magnitude_intervals.value,
        producer_dtype=source.producer_dtype,
        grid_precision=source.grid_precision,
    )
    # This query is inside all ranges but needs the missing grid point.
    with pytest.raises(ValueError, match="unmeasured grid points"):
        masked.predict(
            positions=[[-0.5, -0.5]] * u.arcsec,
            magnitudes=9 * u.mag,
            zenith_angle=20 * u.deg,
        )
    masked.predict(
        positions=[[1, 1]] * u.arcsec,
        magnitudes=10 * u.mag,
        zenith_angle=20 * u.deg,
    )
