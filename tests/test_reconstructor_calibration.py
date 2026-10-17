"""Shared reconstructor calibration without any E2E files or detector model."""

import json
import subprocess
import sys
from contextlib import contextmanager

import numpy as np
import pytest
from astropy import units as u

from hybrid_ao_psf import (
    E2ECalibration,
    E2ECalibrationConfiguration,
    ReconstructorCalibrationSettings,
    ReconstructorNoiseMeasurements,
    build_reconstructor_calibration,
    load_reconstructor_calibration,
    load_reconstructor_measurements,
    save_reconstructor_calibration,
    save_reconstructor_measurements,
)


class SyntheticCalibration(E2ECalibration):
    def measure_reconstructor_noise(self, zenith_angle, inputs):
        assert zenith_angle == 20 * u.deg
        variance = (
            0.001
            * inputs.fwhm.to_value(u.mas) ** 2
            / (inputs.photons_per_frame.to_value(u.photon) * inputs.ee.value)
        )
        return ReconstructorNoiseMeasurements(
            covariance=variance[:, None, None] * np.eye(2) * u.mas**2, dtype=np.float64
        )

    def controller(self, zenith_angle):
        raise NotImplementedError

    def photometry(self, zenith_angle):
        raise NotImplementedError

    def read_covariance_telemetry(self, zenith_angles=None):
        raise NotImplementedError

    def read_coma_wavefronts(self, zenith_angle, samples):
        raise NotImplementedError

    def measure_coma_wfs(self, zenith_angle, wavefronts, sampling):
        raise NotImplementedError

    def measure_sensing_wfs(self, zenith_angle, spot, sampling):
        raise NotImplementedError


@pytest.fixture
def calibration(tmp_path):
    config = E2ECalibrationConfiguration(
        source=tmp_path / "calibration.ini",
        covariance_model=tmp_path / "covariance/model.json",
        coma_model=tmp_path / "coma/model.json",
        sensing_model=tmp_path / "sensing/model.json",
        reconstructor_model=tmp_path / "reconstructor/model.json",
        science_psf=tmp_path / "absent/science.pkl",
        ngs_psf=tmp_path / "absent/ngs.pkl",
        ngs_metrics=tmp_path / "absent/metrics.pkl",
        reconstructor=ReconstructorCalibrationSettings(holdouts=11),
        dtype=np.float64,
    )
    return SyntheticCalibration(config)


@pytest.mark.parametrize("separate", [False, True])
def test_measure_offline_build_roundtrip_and_support(calibration, tmp_path, separate):
    measured = calibration.measure_reconstructor(
        20 * u.deg,
        photons_per_frame=np.geomspace(1, 1e4, 5) * u.photon,
        fwhm=np.geomspace(0.08, 0.32, 4) * u.arcsec,
        ee=np.array([0.2, 0.5, 1]) * u.one if separate else None,
    )
    measurement = tmp_path / "measurements.npz"
    save_reconstructor_measurements(measured, measurement)
    restored = load_reconstructor_measurements(measurement)
    np.testing.assert_array_equal(restored.covariance, measured.covariance)
    built = build_reconstructor_calibration(restored, dtype=np.float64)
    assert built.diagnostics["product_reduction_verified"]
    if not separate:
        assert built.diagnostics["holdout_max_abs_percent"] < 1e-10
    path = tmp_path / "model/model.json"
    save_reconstructor_calibration(built.model, path)
    model = load_reconstructor_calibration(path)
    count = np.array([100, 1000]) * u.photon
    fraction = np.array([0.5, 0.8]) * u.one
    width = np.array([100, 120]) * u.mas
    result = model.predict(photons_per_frame=count, ee=fraction, fwhm=width)
    inverse_ee = (
        np.exp(np.interp(fraction.value, [0.2, 0.5, 1.0], -np.log([0.2, 0.5, 1.0])))
        if separate
        else 1 / fraction.value
    )
    expected = 0.001 * width.value**2 / count.value * inverse_ee
    np.testing.assert_allclose(
        result.to_value(u.mas**2),
        expected[:, None, None] * np.eye(2),
        rtol=1e-14,
        atol=1e-14,
    )
    np.testing.assert_allclose(
        model.measurement_noise(photons_per_frame=count, ee=fraction, fwhm=width).value,
        np.diag(np.repeat(expected, 2)),
        rtol=1e-14,
    )
    with pytest.raises(ValueError, match="support"):
        model.predict(
            photons_per_frame=1e10 * u.photon, ee=0.5 * u.one, fwhm=100 * u.mas
        )
    with pytest.raises(ValueError):
        model.variance.value.flags.writeable = True
    with pytest.raises(TypeError):
        built.diagnostics["holdout_count"] = 0


def test_interrupted_manifest_publication_preserves_accepted_prior(
    calibration, tmp_path, monkeypatch
):
    import hybrid_ao_psf._calibration_products as products

    measured = calibration.measure_reconstructor(
        20 * u.deg, photons_per_frame=[1, 1000] * u.photon, fwhm=[80, 200] * u.mas
    )
    model = build_reconstructor_calibration(measured, dtype=np.float64).model
    path = tmp_path / "model.json"
    save_reconstructor_calibration(model, path)
    before = path.read_bytes()
    original = products.atomic_output_path

    @contextmanager
    def interrupt(destination):
        if destination.name == "model.json":
            raise KeyboardInterrupt
        with original(destination) as temporary:
            yield temporary

    monkeypatch.setattr(products, "atomic_output_path", interrupt)
    with pytest.raises(KeyboardInterrupt):
        save_reconstructor_calibration(model, path)
    assert path.read_bytes() == before
    np.testing.assert_array_equal(
        load_reconstructor_calibration(path).variance, model.variance
    )


def test_wrong_product_units_are_rejected(calibration, tmp_path):
    measured = calibration.measure_reconstructor(
        20 * u.deg, photons_per_frame=[1, 1000] * u.photon, fwhm=[80, 200] * u.mas
    )
    path = tmp_path / "model.json"
    save_reconstructor_calibration(
        build_reconstructor_calibration(measured, dtype=np.float64).model, path
    )
    header = json.loads(path.read_text())
    header["units"]["variance"] = "arcsec2"
    path.write_text(json.dumps(header))
    with pytest.raises(ValueError, match="units"):
        load_reconstructor_calibration(path)


def test_offline_build_and_evaluation_with_native_imports_blocked(
    calibration, tmp_path
):
    measured = calibration.measure_reconstructor(
        20 * u.deg, photons_per_frame=[1, 1000] * u.photon, fwhm=[80, 200] * u.mas
    )
    path = tmp_path / "measurements.npz"
    save_reconstructor_measurements(measured, path)
    script = """
import importlib.abc,sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path,target=None):
        if fullname.split('.')[0] in {'ao_tools','girmos_oomao','matlab'}:
            raise AssertionError('native/downstream import: '+fullname)
sys.meta_path.insert(0,Block())
from hybrid_ao_psf import load_reconstructor_measurements,build_reconstructor_calibration
result=build_reconstructor_calibration(load_reconstructor_measurements(sys.argv[1]))
assert result.diagnostics['holdout_count']==11
"""
    subprocess.run([sys.executable, "-c", script, str(path)], check=True)


def test_public_explicit_covariance_solver_preserves_absolute_weights_and_units():
    from hybrid_ao_psf import solve_lo_reconstructor

    css = np.diag([3.0, 4.0, 5.0, 6.0])
    noise = np.diag([1.0, 2.0, 3.0, 4.0])
    cas = np.array([[1.0, 0.0, 0.5, 0.0], [0.0, 2.0, 0.0, 0.7]])
    weights = solve_lo_reconstructor(
        css * u.mas**2, cas * u.mas**2, noise * u.mas**2, dtype=np.float64
    )
    np.testing.assert_array_equal(weights.value, cas / np.diag(css + noise))
    assert weights.unit == u.one
    with pytest.raises(ValueError):
        weights.value.flags.writeable = True
    with pytest.raises(TypeError):
        solve_lo_reconstructor(css, cas * u.mas**2, noise * u.mas**2, dtype=np.float64)
