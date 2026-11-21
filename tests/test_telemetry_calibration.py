"""Check the file-backed calibration interface with independent HDF5 inputs."""

from pathlib import Path

import h5py
import numpy as np
import pytest
from astropy import units as u

from hybrid_ao_psf import (
    CalibrationPhotometry,
    E2ECalibrationConfiguration,
    FixedGainIntegrator,
    SensingCalibrationSettings,
    TelemetryCalibration,
    ZenithTelemetry,
    load_reconstructor_measurements,
    load_sensing_measurements,
    load_telemetry_calibration,
    save_reconstructor_measurements,
    save_sensing_measurements,
)


def _calibration(
    directory: Path,
    planes: list[ZenithTelemetry],
    *,
    dtype=np.float64,
    block_draws=4,
    asterism_counts=None,
) -> TelemetryCalibration:
    return TelemetryCalibration(
        E2ECalibrationConfiguration(
            source=directory / "calibrate.ini",
            covariance_model="covariance/model.json",
            reconstructor_model="reconstructor/model.json",
            coma_model="coma/model.json",
            sensing_model="sensing/model.json",
            sensing=SensingCalibrationSettings(block_draws=block_draws),
            dtype=dtype,
        ),
        reconstructor_path=directory / "centroid-noise.h5",
        planes=planes,
        controller=FixedGainIntegrator(gain=0.3, delay=2, frame_rate=500 * u.Hz),
        photometry=CalibrationPhotometry(
            magnitude_band="test", photons_per_frame_zeropoint=1e8 * u.photon
        ),
        sensing_wavelength=710 * u.nm,
        field_radius=60 * u.arcsec,
        obscuration_ratio=0.16,
        asterism_counts=asterism_counts,
    )


def test_coma_reads_each_files_sampling_without_opening_other_families(tmp_path):
    planes = []
    for angle, samples, draws in ((20, 4, 8), (30, 8, 12)):
        path = tmp_path / f"probe-{angle}.h5"
        with h5py.File(path, "w") as telemetry:
            telemetry["wavefront"] = np.zeros((samples, 5, 5))
            telemetry["coma"] = np.zeros((samples, 2))
            telemetry["magnitudes"] = [14.0, 16.0]
            telemetry["tilt"] = np.full((samples, 2, draws, 2), angle)
            telemetry["tilt_noiseless"] = np.zeros((samples, 2, 2))
        planes.append(
            ZenithTelemetry(
                zenith_angle=angle * u.deg,
                covariance_path=tmp_path / f"unavailable-covariance-{angle}.h5",
                coma_path=path,
                sensing_path=tmp_path / f"unavailable-sensing-{angle}.h5",
            )
        )

    first, second = _calibration(tmp_path, planes).measure_coma()
    assert (first.settings.samples, first.settings.noise_draws) == (4, 8)
    assert (second.settings.samples, second.settings.noise_draws) == (8, 12)
    np.testing.assert_array_equal(first.response.tilt_batches.value, 20)
    np.testing.assert_array_equal(second.response.tilt_batches.value, 30)


def test_sensing_infers_draws_and_orders_shifts_from_the_file(tmp_path):
    path = tmp_path / "detector-draws.h5"
    shifts = [[0, -7], [7, 0], [0, 7], [-7, 0], [3, 3]]
    with h5py.File(path, "w") as telemetry:
        telemetry["positions"] = [[0, 0]]
        telemetry["magnitudes"] = [14.0]
        telemetry["ee"] = [0.5]
        telemetry["fwhm"] = [100.0]
        telemetry["shifts"] = shifts
        telemetry["tilt_unshifted"] = np.zeros((1, 1, 8, 2))
        telemetry["tilt_shifted"] = np.broadcast_to(
            np.array(shifts)[None, None, :, None, :], (1, 1, 5, 16, 2)
        )
    plane = ZenithTelemetry(
        zenith_angle=20 * u.deg,
        covariance_path=tmp_path / "unavailable-covariance.h5",
        coma_path=tmp_path / "unavailable-coma.h5",
        sensing_path=path,
    )

    [measurement] = _calibration(tmp_path, [plane]).read_sensing()
    assert measurement.displacement == 7 * u.mas
    assert measurement.sampling.zero_draws == 8
    assert measurement.sampling.gain_draws == 16
    np.testing.assert_array_equal(
        measurement.samples.shifted.value[:, 0], [[7, 0], [-7, 0], [0, 7], [0, -7]]
    )


def test_ini_loader_resolves_zenith_template_relative_to_ini(tmp_path):
    directory = tmp_path / "config"
    telemetry_dir = directory / "captures"
    telemetry_dir.mkdir(parents=True)
    pupil = np.eye(5, dtype=bool)
    with h5py.File(telemetry_dir / "pupil.h5", "w") as saved:
        saved["pupil"] = pupil
    with h5py.File(telemetry_dir / "probe-22.5.h5", "w") as telemetry:
        telemetry["wavefront"] = np.broadcast_to(pupil.astype(float), (4, 5, 5))
        telemetry["coma"] = np.zeros((4, 2))
        telemetry["magnitudes"] = [14.0, 16.0]
        telemetry["tilt"] = np.ones((4, 2, 8, 2))
        telemetry["tilt_noiseless"] = np.zeros((4, 2, 2))
    ini = directory / "calibrate.ini"
    ini.write_text(
        "[inputs]\n"
        "ReconstructorTelemetryPath = captures/reconstructor.h5\n"
        "CovarianceTelemetryPath = captures/covariance-{ZA}.h5\n"
        "ComaTelemetryPath = captures/probe-{ZA}.h5\n"
        "SensingTelemetryPath = captures/sensor-{ZA}.h5\n"
        "[outputs]\n"
        "ReconstructorModelPath = products/reconstructor/model.json\n"
        "CovarianceModelPath = products/covariance/model.json\n"
        "ComaModelPath = products/coma/model.json\n"
        "SensingModelPath = products/sensing/model.json\n"
        "[system]\n"
        "ZenithAngle = [22.5]\n"
        "Wavelength = [710e-9]\n"
        "FieldRadius = 60\n"
        "ObscurationRatio = 0.16\n"
        "PupilMaskPath = captures/pupil.h5\n"
        "LoopGain_LO = 0.3\n"
        "LoopDelaySteps_LO = 2\n"
        "SensorFrameRate_LO = 500\n"
        "NGSMagStandard = test\n"
        "NGSMagZeropoint = 1e8\n"
    )

    calibration = load_telemetry_calibration(ini, dtype=np.float64)
    assert (
        calibration.configuration.coma_model == directory / "products/coma/model.json"
    )
    assert calibration.configuration.dtype == np.dtype("float64")
    [measurement] = calibration.measure_coma()
    assert measurement.zenith_angle == 22.5 * u.deg
    assert measurement.sensing_wavelength == 710e-9 * u.m
    assert measurement.photometry.photons_per_frame_zeropoint == 1e8 * u.photon
    ini.write_text(ini.read_text().replace("PupilMaskPath = captures/pupil.h5\n", ""))
    annulus_calibration = load_telemetry_calibration(ini, dtype=np.float64)
    with pytest.raises(ValueError, match="wavefront, coma and pupil must be aligned"):
        next(annulus_calibration.measure_coma())


@pytest.mark.parametrize(
    "producer_dtype, calculation_dtype",
    [
        (np.float64, np.float32),
        (np.float32, np.float64),
    ],
)
def test_reconstructor_uses_captured_probes(
    tmp_path, producer_dtype, calculation_dtype
):
    path = tmp_path / "centroid-noise.h5"
    signal, width, fraction = np.meshgrid(
        np.geomspace(10, 100, 6),
        np.geomspace(60, 140, 4),
        [0.3, 0.65, 0.9],
        indexing="ij",
    )
    with h5py.File(path, "w") as file:
        grid = file.create_group("grid")
        grid["signal"] = np.asarray([10, 100], dtype=producer_dtype)
        grid["fwhm"] = np.asarray([60, 140], dtype=producer_dtype)
        grid["covariance"] = np.broadcast_to(
            np.eye(2, dtype=producer_dtype), (2, 2, 2, 2)
        )
        checks = file.create_group("checks")
        checks["flux"] = np.r_[32, (signal / fraction).ravel()].astype(producer_dtype)
        checks["ee"] = np.r_[0.7, fraction.ravel()].astype(producer_dtype)
        checks["fwhm"] = np.r_[93, width.ravel()].astype(producer_dtype)
        checks["covariance"] = np.broadcast_to(
            np.eye(2, dtype=producer_dtype), (73, 2, 2)
        )

    calibration = _calibration(tmp_path, [], dtype=calculation_dtype)
    measurements = calibration.read_reconstructor_measurements()
    assert measurements.dtype == producer_dtype
    assert measurements.seed is None
    np.testing.assert_array_equal(measurements.inputs.fwhm.value[4], 93)
    retained = tmp_path / "reconstructor-measurements.npz"
    save_reconstructor_measurements(measurements, retained)
    result = calibration.build_reconstructor(load_reconstructor_measurements(retained))
    assert result.model.dtype == calculation_dtype
    assert result.diagnostics["interpolation_check_count"] == 1
    np.testing.assert_array_equal(result.model.variance.value, 1)

    if producer_dtype == np.float32:
        with h5py.File(path, "a") as file:
            del file["grid/covariance"]
            captured = np.diag([1, 1 + 2e-8])
            file["grid/covariance"] = np.broadcast_to(captured, (2, 2, 2, 2))
        mixed = calibration.read_reconstructor_measurements()
        assert mixed.dtype == np.float64
        assert mixed.grid_precision == np.float32
        save_reconstructor_measurements(mixed, retained)
        mixed = load_reconstructor_measurements(retained)
        assert mixed.grid_precision == np.float32
        with pytest.raises(ValueError, match="anisotropic noise"):
            calibration.build_reconstructor(mixed)

        with h5py.File(path, "a") as file:
            del file["grid/signal"]
            del file["grid/fwhm"]
            file["grid/signal"] = [10.0, 100.0]
            file["grid/fwhm"] = [60.0, 140.0]
        assert (
            calibration.read_reconstructor_measurements().grid_precision == np.float64
        )

    with h5py.File(path, "a") as file:
        file["checks/flux"][1] *= 1.1
    with pytest.raises(ValueError, match="equal-signal check"):
        calibration.read_reconstructor_measurements()


def test_rejects_one_file_for_different_zeniths(tmp_path):
    path = tmp_path / "covariance.h5"
    path.touch()
    alias = tmp_path / "covariance-copy.h5"
    alias.symlink_to(path)
    with pytest.raises(ValueError, match="different covariance telemetry file"):
        _calibration(
            tmp_path,
            [
                ZenithTelemetry(
                    zenith_angle=angle * u.deg,
                    covariance_path=source,
                    coma_path=tmp_path / f"coma-{angle}.h5",
                    sensing_path=tmp_path / f"sensing-{angle}.h5",
                )
                for angle, source in ((20, path), (30, alias))
            ],
        )


def test_sensing_uses_measured_directions_and_retained_spot_metrics(tmp_path):
    rng = np.random.default_rng(17)
    planes = []
    positions_by_angle = {
        20: [(0, 0), (2, 3), (5, 4)],
        30: [(3, 2), (6, 4), (8, 7)],
    }
    shifts = np.array([[8, 0], [-8, 0], [0, 8], [0, -8]])
    for angle, positions in positions_by_angle.items():
        path = tmp_path / f"sensing-{angle}.h5"
        with h5py.File(path, "w") as file:
            file["positions"] = positions
            file["magnitudes"] = [14.0, 16.0]
            file["ee"] = [0.3, 0.4, 0.5]
            file["fwhm"] = [100.0, 200.0, 300.0]
            file["shifts"] = shifts
            noise = rng.standard_normal((3, 2, 120, 2))
            file["tilt_unshifted"] = noise
            file["tilt_shifted"] = (
                noise[:, :, None] + 0.9 * shifts[None, None, :, None, :]
            )
        planes.append(
            ZenithTelemetry(
                zenith_angle=angle * u.deg,
                covariance_path=tmp_path / f"covariance-{angle}.h5",
                coma_path=tmp_path / f"coma-{angle}.h5",
                sensing_path=path,
            )
        )
    calibration = _calibration(
        tmp_path, planes, block_draws=6, asterism_counts=(1, 1, 0)
    )
    measurements = list(calibration.read_sensing())
    saved = tmp_path / "sensing-measurements.npz"
    save_sensing_measurements(measurements[0], saved)
    measurements[0] = load_sensing_measurements(saved)
    response = calibration.build_sensing_response(measurements)
    for plane in planes:
        plane.sensing_path.unlink()

    cases = list(calibration.sensing_noise_requests(response))
    assert len(cases) == 4
    for case in cases:
        positions = positions_by_angle[int(case.zenith_angle.to_value(u.deg))]
        idx = [positions.index(tuple(point)) for point in case.positions.value]
        np.testing.assert_array_equal(case.ee.value, np.array([0.3, 0.4, 0.5])[idx])
        np.testing.assert_array_equal(
            case.fwhm.value, np.array([100.0, 200.0, 300.0])[idx]
        )
