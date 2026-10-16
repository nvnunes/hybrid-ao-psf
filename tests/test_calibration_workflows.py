"""Complete configured calibration handoff using native-independent telemetry."""

from contextlib import contextmanager
from dataclasses import replace

import numpy as np
import pytest
from astropy import units as u

from hybrid_ao_psf import (
    CalibrationPhotometry,
    ComaCalibrationSettings,
    ComaWavefronts,
    ComaWfsMeasurements,
    CovarianceCalibrationSettings,
    CovarianceTelemetry,
    E2ECalibration,
    E2ECalibrationConfiguration,
    FixedGainIntegrator,
    ReconstructorCalibrationSettings,
    ReconstructorNoiseMeasurements,
    ScienceHoPsfSamples,
    SensingCalibrationSettings,
    SensingReplayRequest,
    SensingWfsMeasurements,
    build_science_ho_psf_interpolator,
    build_sensing_response,
    load_coma_calibration,
    load_covariance_calibration,
    load_reconstructor_calibration,
    load_sensing_calibration,
    save_coma_calibration,
    save_covariance_calibration,
    save_reconstructor_calibration,
    save_science_ho_psf_interpolator,
    save_sensing_calibration,
)


class SyntheticCalibration(E2ECalibration):
    native_available = True

    def _native(self):
        if not self.native_available:
            raise AssertionError("offline operation accessed native hooks")

    def controller(self, zenith_angle):
        self._native()
        return FixedGainIntegrator(gain=0.3, delay=2, frame_rate=500 * u.Hz)

    def photometry(self, zenith_angle):
        self._native()
        return CalibrationPhotometry(
            magnitude_band="fixture", photons_per_frame_zeropoint=1e8 * u.photon
        )

    def measure_reconstructor_noise(self, zenith_angle, inputs):
        self._native()
        variance = (
            0.001
            * inputs.fwhm.value**2
            / (inputs.photons_per_frame.value * inputs.ee.value)
        )
        return ReconstructorNoiseMeasurements(
            covariance=variance[:, None, None] * np.eye(2) * u.mas**2
        )

    def read_covariance_telemetry(self, zenith_angles=None):
        self._native()
        positions = np.array([[-1, -1], [-1, 1], [1, -1], [1, 1]])
        for angle in (20, 40):
            rng = np.random.default_rng(70 + angle)
            data = rng.normal(size=(3, 600, 4, 2))
            data += np.arange(3)[:, None, None, None]
            yield CovarianceTelemetry(
                zenith_angle=angle * u.deg,
                positions=positions * u.arcsec,
                frame_rate=500 * u.Hz,
                tilt=data * u.mas,
                coma=data * 2 * u.nm,
            )

    def read_coma_wavefronts(self, zenith_angle, samples):
        self._native()
        rng = np.random.default_rng(10 + int(zenith_angle.value))
        return ComaWavefronts(
            wavefront=rng.normal(size=(samples, 4, 4)) * u.nm,
            coma=(rng.normal(size=(samples, 2)) * 30 + 4) * u.nm,
            pupil=np.ones((4, 4), dtype=bool),
        )

    def measure_coma_wfs(self, zenith_angle, wavefronts, sampling):
        self._native()
        response = (0.1 + 0.04 * (1 / np.cos(zenith_angle.to_value(u.rad)) - 1)) / (
            1 + 10 ** (0.4 * (sampling.magnitudes.value - 18.5))
        )
        tilt = wavefronts.coma.value[:, None, :] * response[None, :, None]
        # Paired independent batch means average to the exact local response.
        rng = np.random.default_rng(57)
        noise = rng.normal(size=(len(tilt), len(response), 4, 2)) * 0.02
        noise -= noise.mean(axis=2, keepdims=True)
        draws = np.repeat(
            tilt[:, :, None, :] + noise, sampling.noise_draws // 4, axis=2
        )
        return ComaWfsMeasurements(measured=draws * u.mas, noiseless=tilt * u.mas)

    def measure_sensing_wfs(self, zenith_angle, spot, sampling):
        self._native()
        assert np.sum(spot.image) == pytest.approx(0.5)
        assert spot.pixel_scale == 2 * u.mas
        rng = np.random.default_rng(sampling.seed)
        zero = rng.normal(size=(sampling.zero_draws, 2)) + [3, -7]
        paired = rng.normal(size=(sampling.gain_draws, 2))
        shifted = paired[None, :, :] + spot.displacements.value[1:, None, :]
        return SensingWfsMeasurements(zero=zero * u.mas, shifted=shifted * u.mas)


@pytest.fixture(scope="module")
def pipeline(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("calibration-pipeline")
    config = E2ECalibrationConfiguration(
        source=tmp_path / "unused.ini",
        **{
            f"{family}_model": tmp_path / family / "model.json"
            for family in ("reconstructor", "covariance", "coma", "sensing")
        },
        science_psf=tmp_path / "unused-science.pkl",
        ngs_psf=tmp_path / "spots.pkl",
        ngs_metrics=tmp_path / "unused-metrics.pkl",
        reconstructor=ReconstructorCalibrationSettings(holdouts=7),
        covariance=CovarianceCalibrationSettings(
            feedback=(0.5, 1, 1.5), bootstrap_draws=3
        ),
        coma=ComaCalibrationSettings(samples=32, noise_draws=16),
        sensing=SensingCalibrationSettings(
            zero_draws=128,
            gain_draws=64,
            block_draws=8,
            replay_realizations=3,
            replay_frames=256,
            discard_frames=96,
        ),
    )
    positions = np.array([[-1, -1], [-1, 1], [1, -1], [1, 1]])
    save_science_ho_psf_interpolator(
        build_science_ho_psf_interpolator(
            ScienceHoPsfSamples(
                x=positions[:, 0] * u.arcsec,
                y=positions[:, 1] * u.arcsec,
                psfs=np.full((4, 4, 4), 0.5 / 16),
                wavelength=710 * u.nm,
                pixel_scale=2 * u.mas,
                tel_diameter=8 * u.m,
                tel_pupil=np.ones((4, 4)) * u.one,
                zenith_angle=20 * u.deg,
            )
        ),
        config.ngs_psf,
    )
    calibration = SyntheticCalibration(config)
    prior = calibration.measure_reconstructor(
        20 * u.deg, photons_per_frame=[1, 1e8] * u.photon, fwhm=[80, 200] * u.mas
    )
    calibration.native_available = False
    prior = calibration.build_reconstructor(prior).model
    calibration.native_available = True
    covariances = list(calibration.measure_covariance())
    telemetry = list(calibration.read_covariance_telemetry())
    calibration.native_available = False
    covariances = calibration.build_covariance(
        covariances, validation_telemetry=telemetry
    ).model
    calibration.native_available = True
    coma = [
        calibration.measure_coma(
            angle * u.deg,
            np.arange(8.0, 21.0, 2) * u.mag,
            sensing_wavelength=710e-9 * u.m,
        )
        for angle in (0, 20, 30, 40)
    ]
    calibration.native_available = False
    coma = calibration.build_coma(coma).model
    calibration.native_available = True
    rows = np.array([[x, y, m] for x in (-1, 1) for y in (-1, 1) for m in (8, 10)])
    points = list(
        calibration.measure_sensing(
            20 * u.deg,
            positions=rows[:, :2] * u.arcsec,
            magnitudes=rows[:, 2] * u.mag,
            wavelength=710 * u.nm,
            seeds=range(80, 88),
        )
    )
    calibration.native_available = False
    response = build_sensing_response(
        points, field_radius=2 * u.arcsec, magnitude_intervals=[[8, 10]] * u.mag
    )
    requests = [
        SensingReplayRequest(
            key=str(n),
            population="synthetic",
            zenith_angle=20 * u.deg,
            positions=np.array([[0, 0], [0.3, -0.2], [-0.4, 0.5]])[:n] * u.arcsec,
            magnitudes=np.full(n, 9) * u.mag,
            ee=np.full(n, 0.5) * u.one,
            fwhm=np.full(n, 100) * u.mas,
            seed=900 + n,
        )
        for n in (1, 2, 3)
    ]
    replays = list(calibration.replay_sensing(response, requests))
    sensing = calibration.build_sensing(response, replays).model
    return calibration, (prior, covariances, coma, sensing)


def test_configured_four_family_handoff_without_native_files(pipeline):
    calibration, models = pipeline
    config = calibration.configuration
    loaded = (
        load_reconstructor_calibration(config.reconstructor_model),
        load_covariance_calibration(config.covariance_model),
        load_coma_calibration(config.coma_model),
        load_sensing_calibration(config.sensing_model),
    )
    np.testing.assert_array_equal(loaded[0].variance, models[0].variance)
    np.testing.assert_array_equal(
        loaded[1].normalized_tables["C_t_t"], models[1].normalized_tables["C_t_t"]
    )
    assert loaded[2].a0 == models[2].a0
    assert loaded[3].blur.parameters() == models[3].blur.parameters()
    for family in ("reconstructor", "covariance", "coma", "sensing"):
        assert getattr(config, f"{family}_model").with_name("diagnostics.json").exists()


@pytest.mark.parametrize(
    "family,boundary",
    [
        (family, boundary)
        for family in range(4)
        for boundary in ("payload", "manifest", "committed")
        if not (family == 2 and boundary == "payload")
    ],
)
def test_publication_interruption_leaves_complete_usable_product(
    pipeline, monkeypatch, family, boundary, tmp_path
):
    import hybrid_ao_psf._calibration_products as products

    _, models = pipeline
    model = models[family]
    save = (
        save_reconstructor_calibration,
        save_covariance_calibration,
        save_coma_calibration,
        save_sensing_calibration,
    )[family]
    load = (
        load_reconstructor_calibration,
        load_covariance_calibration,
        load_coma_calibration,
        load_sensing_calibration,
    )[family]
    path = tmp_path / "published/model.json"
    save(model, path)
    path.chmod(0o640)
    before = path.read_bytes()
    original = products.atomic_output_path

    @contextmanager
    def interrupted(destination):
        if boundary == "payload" and destination.suffix == ".npz":
            raise KeyboardInterrupt
        if destination.name == "model.json" and boundary == "manifest":
            with original(destination) as temporary:
                yield temporary
                raise KeyboardInterrupt
        else:
            with original(destination) as temporary:
                yield temporary
            if destination.name == "model.json" and boundary == "committed":
                raise KeyboardInterrupt

    if family == 0:
        model = replace(model, variance=model.variance * 1.01)
    elif family == 1:
        arrays = {k: v.copy() for k, v in model._bank.arrays.items()}
        for name in ("C_t_t", "C_Tc_Tc", "C_t_Tc", "C_Tt_Tc", "C_Tt_Tt", "C_t_Tt"):
            arrays[name] *= 1.01
        model = type(model)._from_arrays(arrays, model.controller, model.discard)
    elif family == 2:
        model = replace(model, a0=model.a0 * 1.01)
    else:
        response = model.response
        arrays = {key: value.copy() for key, value in response._arrays.items()}
        arrays["gain"] *= 1.01
        response = type(response)._from_arrays(
            arrays,
            response.controller,
            response.photometry,
            response.field_radius,
            response.magnitude_intervals.value,
        )
        model = replace(model, response=response)
    monkeypatch.setattr(products, "atomic_output_path", interrupted)
    with pytest.raises(KeyboardInterrupt):
        save(model, path)
    load(path)
    assert path.stat().st_mode & 0o777 == 0o640
    assert path.read_bytes() == before


def test_calibrated_reconstructor_validation_rejects_mismatched_telemetry(pipeline):
    from hybrid_ao_psf import validate_covariance_reconstructor

    calibration, models = pipeline
    calibration.native_available = True
    telemetry = next(calibration.read_covariance_telemetry())
    calibration.native_available = False
    changed = replace(telemetry, tilt=telemetry.tilt * 1.1)
    with pytest.raises(ValueError, match="calibrated-reconstructor validation failed"):
        validate_covariance_reconstructor(models[1], models[0], changed)


@pytest.mark.parametrize("family", range(4))
def test_each_product_rejects_wrong_declared_units(pipeline, tmp_path, family):
    import json

    _, models = pipeline
    save = (
        save_reconstructor_calibration,
        save_covariance_calibration,
        save_coma_calibration,
        save_sensing_calibration,
    )[family]
    load = (
        load_reconstructor_calibration,
        load_covariance_calibration,
        load_coma_calibration,
        load_sensing_calibration,
    )[family]
    path = tmp_path / "model.json"
    save(models[family], path)
    header = json.loads(path.read_text())
    header["units"] = {"wrong": "units"}
    path.write_text(json.dumps(header))
    with pytest.raises(ValueError, match="units"):
        load(path)


def test_configured_publication_restores_model_and_diagnostics_on_cancellation(
    pipeline, monkeypatch
):
    import hybrid_ao_psf._calibration_products as products

    calibration, _ = pipeline
    path = calibration.configuration.reconstructor_model
    diagnostics = path.with_name("diagnostics.json")
    before = (path.read_bytes(), diagnostics.read_bytes())
    calibration.native_available = True
    observations = calibration.measure_reconstructor(
        20 * u.deg, photons_per_frame=[1, 1e8] * u.photon, fwhm=[80, 220] * u.mas
    )
    calibration.native_available = False
    original = products.atomic_output_path

    @contextmanager
    def interrupted(destination):
        with original(destination) as temporary:
            yield temporary
        if destination == path:
            raise KeyboardInterrupt

    monkeypatch.setattr(products, "atomic_output_path", interrupted)
    with pytest.raises(KeyboardInterrupt):
        calibration.build_reconstructor(observations)
    assert (path.read_bytes(), diagnostics.read_bytes()) == before
    load_reconstructor_calibration(path)


def test_shared_model_destinations_cannot_overwrite_another_family(pipeline):
    calibration, _ = pipeline
    with pytest.raises(ValueError, match="destinations must be distinct"):
        replace(
            calibration.configuration,
            coma_model=calibration.configuration.sensing_model,
        )


@pytest.mark.parametrize("wavelength", [-1 * u.m, [1, 2] * u.m, 1 * u.s])
def test_invalid_coma_wavelength_does_not_dispatch_native(pipeline, wavelength):
    calibration = SyntheticCalibration(pipeline[0].configuration)
    calibration.native_available = False
    with pytest.raises((ValueError, TypeError, u.UnitConversionError)):
        calibration.measure_coma(
            20 * u.deg, [8, 10] * u.mag, sensing_wavelength=wavelength
        )


@pytest.mark.parametrize(
    "updates",
    [
        {"position": [1] * u.arcsec},
        {"position": [1, 2] * u.s},
        {"magnitude": [8, 9] * u.mag},
        {"magnitude": 8 * u.s},
        {"sampling": {}},
    ],
)
def test_invalid_sensing_arguments_do_not_dispatch_native(pipeline, updates):
    from hybrid_ao_psf import SensingSpot

    calibration = SyntheticCalibration(pipeline[0].configuration)
    calibration.native_available = False
    spot = SensingSpot(
        image=np.ones((4, 4)),
        pixel_scale=2 * u.mas,
        wavelength=710 * u.nm,
        photons_per_frame=1 * u.photon,
        displacements=np.array([[0, 0], [10, 0], [-10, 0], [0, 10], [0, -10]]) * u.mas,
    )
    values = {"position": [0, 0] * u.arcsec, "magnitude": 8 * u.mag, "sampling": None}
    values.update(updates)
    with pytest.raises((ValueError, TypeError, u.UnitConversionError)):
        calibration.measure_sensing_point(20 * u.deg, spot=spot, **values)
