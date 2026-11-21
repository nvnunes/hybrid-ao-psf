"""Independent physical-input, configuration, persistence and file checks."""

import json
from dataclasses import FrozenInstanceError

import numpy as np
import pytest
from astropy import units as u

from hybrid_ao_psf import (
    CalibrationModelFiles,
    CalibrationPhotometry,
    ComaResponseMeasurements,
    ComaWavefronts,
    ComaWfsMeasurements,
    CovarianceTelemetry,
    E2ECalibration,
    FixedGainIntegrator,
    ReconstructorNoiseInputs,
    ReconstructorNoiseMeasurements,
    SensingSpot,
    SensingWfsMeasurements,
    load_calibration_data,
    load_e2e_calibration_configuration,
    read_calibration_model_files,
    save_calibration_data,
)


def telemetry():
    values = np.arange(2 * 5 * 3 * 2).reshape(2, 5, 3, 2).astype(float)
    return CovarianceTelemetry(
        zenith_angle=20 * u.deg,
        positions=[[0, 0], [1, 0], [0, 1]] * u.arcsec,
        frame_rate=500 * u.Hz,
        tilt=values * u.mas,
        coma=(values / 10) * u.nm,
    )


def ini(tmp_path, extra=""):
    path = tmp_path / "shared.ini"
    path.write_text(
        "[e2e]\n"
        + "".join(
            f"{name}_model = products/{name}/model.json\n"
            for name in ("covariance", "coma", "sensing", "reconstructor")
        )
        + "science_psf = missing/science.pkl\nngs_psf = missing/ngs.pkl\nngs_metrics = missing/metrics.pkl\n"
        + extra
    )
    return path


def test_reconstructor_point_counts_in_ini_and_overrides(tmp_path):
    path = ini(
        tmp_path,
        "[calibration.reconstructor]\n"
        "signal_point_count = 5\nfwhm_point_count = 7\nee_point_count = 3\n",
    )
    config = load_e2e_calibration_configuration(path)
    assert config.reconstructor.signal_point_count == 5
    assert config.reconstructor.fwhm_point_count == 7
    assert config.reconstructor.ee_point_count == 3
    changed = config.with_overrides(reconstructor={"ee_point_count": 4})
    assert changed.reconstructor.ee_point_count == 4

    path = ini(tmp_path, "[calibration.reconstructor]\nsignal_nodes = 5\n")
    with pytest.raises(ValueError, match="Unknown keys.*signal_nodes"):
        load_e2e_calibration_configuration(path)


@pytest.mark.parametrize("setting", ["holdouts", "seed"])
def test_ini_rejects_reconstructor_interpolation_check_options(tmp_path, setting):
    path = ini(tmp_path, f"[calibration.reconstructor]\n{setting} = 11\n")
    with pytest.raises(ValueError, match=f"Unknown keys.*{setting}"):
        load_e2e_calibration_configuration(path)


@pytest.mark.parametrize("setting", ["bootstrap_draws", "seed"])
def test_ini_rejects_covariance_bootstrap_options(tmp_path, setting):
    path = ini(tmp_path, f"[calibration.covariance]\n{setting} = 11\n")
    with pytest.raises(ValueError, match=f"Unknown keys.*{setting}"):
        load_e2e_calibration_configuration(path)


def test_ini_rejects_coma_bootstrap_seed(tmp_path):
    path = ini(tmp_path, "[calibration.coma]\nbootstrap_seed = 57\n")
    with pytest.raises(ValueError, match="Unknown keys.*bootstrap_seed"):
        load_e2e_calibration_configuration(path)


def test_packed_independent_realizations_and_immutable_unit_state():
    result = telemetry()
    expected = np.arange(60).reshape(2, 5, 3, 2)
    np.testing.assert_array_equal(result.tilt.to_value(u.mas), expected)
    # Pooled centring retains between-realization displacement variance.
    pooled = np.cov(expected.reshape(-1, 6), rowvar=False, bias=True)
    within = np.mean(
        [np.cov(v.reshape(-1, 6), rowvar=False, bias=True) for v in expected], axis=0
    )
    assert np.trace(pooled) > np.trace(within)
    view = result.tilt
    view._unit = u.rad
    assert result.tilt.unit == u.mas
    with pytest.raises(ValueError):
        result.tilt.value[0, 0, 0, 0] = 1
    with pytest.raises(ValueError):
        result.tilt.flags.writeable = True
    with pytest.raises(FrozenInstanceError):
        result.tilt = np.zeros_like(expected) * u.mas


def test_input_conversion_is_owned_and_means_preserved():
    data = np.ones((2, 5, 3, 2))
    record = CovarianceTelemetry(
        zenith_angle=(np.pi / 9) * u.rad,
        positions=[[0, 0], [1, 0], [0, 1]] * u.deg,
        frame_rate=0.5 * u.kHz,
        tilt=data * u.arcsec,
        coma=data * u.um,
    )
    data[:] = 90
    np.testing.assert_allclose(record.tilt.mean().to_value(u.mas), 1000, rtol=1e-15)
    np.testing.assert_allclose(record.coma.mean().to_value(u.nm), 1000, rtol=1e-15)
    assert record.frame_rate == 500 * u.Hz
    np.testing.assert_allclose(record.positions.value[1], [3600, 0])


def test_rejects_unitless_misaligned_or_duplicate_telemetry():
    r = telemetry()
    kwargs = {name: getattr(r, name) for name in r.__dataclass_fields__}
    for name, value in [
        ("tilt", r.tilt.value),
        ("coma", r.coma[:, :-1]),
        ("positions", [[0, 0], [0, 0], [1, 2]] * u.arcsec),
        ("frame_rate", 0 * u.Hz),
    ]:
        with pytest.raises((ValueError, TypeError)):
            CovarianceTelemetry(**{**kwargs, name: value})


def records():
    return [
        telemetry(),
        ComaWavefronts(
            wavefront=np.ones((4, 2, 2)) * u.nm,
            coma=np.ones((4, 2)) * u.nm,
            pupil=np.ones((2, 2), bool),
        ),
        ComaWfsMeasurements(
            measured=np.ones((4, 2, 4, 2)) * u.mas, noiseless=np.ones((4, 2, 2)) * u.mas
        ),
        ComaResponseMeasurements(
            coma=np.ones((4, 2)) * u.nm,
            magnitudes=[8, 19] * u.mag,
            tilt_mean=np.ones((4, 2, 2)) * u.mas,
            tilt_batches=np.ones((4, 2, 4, 2)) * u.mas,
            tilt_noiseless=np.ones((4, 2, 2)) * u.mas,
        ),
        SensingSpot(
            image=np.ones((4, 4)) / 16,
            pixel_scale=1 * u.mas,
            wavelength=710 * u.nm,
            photons_per_frame=100 * u.photon,
            displacements=[[0, 0], [10, 0], [-10, 0], [0, 10], [0, -10]] * u.mas,
        ),
        SensingWfsMeasurements(
            zero=np.ones((10, 2)) * u.mas, shifted=np.ones((4, 20, 2)) * u.mas
        ),
        ReconstructorNoiseInputs(
            photons_per_frame=[100, 1000] * u.photon,
            ee=[0.5, 0.8] * u.one,
            fwhm=[80, 100] * u.mas,
        ),
        ReconstructorNoiseMeasurements(
            covariance=np.array([np.eye(2), np.eye(2) * 2]) * u.mas**2
        ),
    ]


@pytest.mark.parametrize("record", records())
def test_calibration_data_roundtrip_without_native_software(tmp_path, record):
    path = tmp_path / "samples.npz"
    save_calibration_data(record, path)
    result = load_calibration_data(path)
    assert type(result) is type(record)
    for name in record.__dataclass_fields__:
        expected = getattr(record, name)
        actual = getattr(result, name)
        if isinstance(expected, u.Quantity):
            actual = actual.to_value(expected.unit)
            expected = expected.value
        np.testing.assert_array_equal(actual, expected)


def test_saved_wrong_unit_declaration_is_rejected(tmp_path):
    path = tmp_path / "sample.npz"
    save_calibration_data(telemetry(), path)
    with np.load(path) as saved:
        arrays = {key: saved[key] for key in saved.files}
    header = json.loads(str(arrays["__header__"]))
    header["units"]["tilt"] = "rad"
    arrays["__header__"] = json.dumps(header)
    np.savez(path, **arrays)
    with pytest.raises(ValueError, match="units"):
        load_calibration_data(path)


def test_configuration_precedence_and_native_sections_are_not_opened(tmp_path):
    path = ini(
        tmp_path,
        "[oomao]\nsource = /absent/native.mat\n[calibration.covariance]\nfeedback = [0.2, 0.7, 2.0]\n[calibration.sensing]\nzero_draws = 8192\n",
    )
    config = load_e2e_calibration_configuration(path)
    assert not config.covariance_model.exists()
    assert config.covariance.feedback == (0.2, 0.7, 2.0)
    assert config.coma.samples == 512
    changed = config.with_overrides(covariance={"feedback": (0.4, 1.4)})
    assert changed.covariance.feedback == (0.4, 1.4)
    assert changed.sensing.zero_draws == 8192
    assert config.with_overrides().covariance.feedback == (0.2, 0.7, 2.0)
    with pytest.raises(TypeError):
        config.with_overrides(covariance={"typo": 12})
    assert config.covariance_model == tmp_path / "products/covariance/model.json"
    assert issubclass(E2ECalibration, object)
    with pytest.raises(TypeError):
        E2ECalibration(config)


def test_owned_section_typo_and_invalid_override_rejected(tmp_path):
    with pytest.raises(ValueError, match="Unknown"):
        load_e2e_calibration_configuration(
            ini(tmp_path, "[calibration.coma]\nnoise_darws = 512\n")
        )
    config = load_e2e_calibration_configuration(ini(tmp_path))
    with pytest.raises(ValueError):
        config.with_overrides(sensing={"zero_draws": 0})


@pytest.mark.parametrize("pixel_value", [1e40, 1e-50])
def test_sensing_spot_rejects_unrepresentable_flux(pixel_value):
    with pytest.raises(ValueError, match="image flux"):
        SensingSpot(
            image=np.full((3, 3), pixel_value),
            pixel_scale=5 * u.mas,
            wavelength=700 * u.nm,
            photons_per_frame=100 * u.photon,
            displacements=np.array([[0, 0], [1, 0], [-1, 0], [0, 1], [0, -1]]) * u.mas,
            dtype=np.float32,
        )


def test_photometry_is_generic_and_preserves_incident_count_convention():
    conversion = CalibrationPhotometry(
        magnitude_band="test band", photons_per_frame_zeropoint=1e8 * u.photon
    )
    np.testing.assert_allclose(
        conversion.magnitude_to_photons_per_frame([0, 5, 10] * u.mag).value,
        [1e8, 1e6, 1e4],
        rtol=1e-15,
    )
    with pytest.raises(ValueError, match="magnitudes must be real"):
        conversion.magnitude_to_photons_per_frame([10 + 2j] * u.mag)
    q = conversion.photons_per_frame_zeropoint
    q._unit = u.electron
    assert conversion.photons_per_frame_zeropoint.unit == u.photon
    with pytest.raises(ValueError):
        FixedGainIntegrator(gain=0.3, delay=True, frame_rate=500 * u.Hz)


def test_model_files_cover_manifest_and_array_file_and_changed_valid_model(tmp_path):
    import hashlib

    array_file = tmp_path / "arrays.npz"
    np.savez(array_file, values=[1.0, 2.0])
    path = tmp_path / "model.json"
    data = {
        "kind": "synthetic",
        "schema_version": 1,
        "array_file": "arrays.npz",
        "array_file_sha256": hashlib.sha256(array_file.read_bytes()).hexdigest(),
        "coefficient": 1.0,
    }
    path.write_text(json.dumps(data))
    model_files = read_calibration_model_files(path)
    assert isinstance(model_files, CalibrationModelFiles)
    assert len(model_files.array_files) == 1
    assert model_files.manifest.path == path.resolve()
    assert not hasattr(model_files, "entrypoint")
    assert not hasattr(model_files, "payloads")
    model_files.verify()
    # Another valid generation at the same path is nevertheless not this binding.
    np.savez(array_file, values=[2.0, 3.0])
    data["array_file_sha256"] = hashlib.sha256(array_file.read_bytes()).hexdigest()
    path.write_text(json.dumps(data))
    replacement = read_calibration_model_files(path)
    assert replacement != model_files
    with pytest.raises(ValueError, match="content changed"):
        model_files.verify()
    replacement.verify()
    array_file.unlink()
    with pytest.raises(FileNotFoundError):
        replacement.verify()


def test_json_only_model_files_and_bad_array_file_paths(tmp_path):
    path = tmp_path / "model.json"
    path.write_text('{"coefficient": 1}')
    model_files = read_calibration_model_files(path)
    assert model_files.array_files == ()
    for value in ("../elsewhere.npz", "/tmp/elsewhere.npz", "not-an-array.json"):
        path.write_text(json.dumps({"array_file": value, "array_file_sha256": "0" * 64}))
        with pytest.raises(ValueError, match="relative NPZ"):
            read_calibration_model_files(path)


def test_atomic_data_write_preserves_previous_on_cancellation(tmp_path, monkeypatch):
    import hybrid_ao_psf._calibration_io as implementation

    path = tmp_path / "samples.npz"
    save_calibration_data(telemetry(), path)
    original = path.read_bytes()

    def cancelled(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(implementation.np, "savez", cancelled)
    with pytest.raises(KeyboardInterrupt):
        save_calibration_data(telemetry(), path)
    assert path.read_bytes() == original
    assert list(tmp_path.iterdir()) == [path]


def test_common_imports_and_records_do_not_require_downstream_native_modules(tmp_path):
    import subprocess
    import sys

    code = """
import importlib.abc
import sys
class NoNative(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname.split('.')[0] in {'girmos_oomao', 'ao_tools', 'matlab'}:
            raise ImportError('native dependency blocked: ' + fullname)
sys.meta_path.insert(0, NoNative())
from hybrid_ao_psf import FixedGainIntegrator, E2ECalibration, CovarianceTelemetry
from astropy import units as u
assert FixedGainIntegrator(gain=.3, delay=2, frame_rate=500*u.Hz).delay == 2
"""
    subprocess.run(
        [sys.executable, "-c", code],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )


def test_precision_configuration_defaults_ini_and_explicit_override(tmp_path):
    path = ini(tmp_path)
    assert load_e2e_calibration_configuration(path).dtype == np.dtype("float32")
    path = ini(tmp_path, "precision = float64\n")
    configured = load_e2e_calibration_configuration(path)
    assert configured.dtype == np.dtype("float64")
    assert configured.with_overrides().dtype == np.dtype("float64")
    assert configured.with_overrides(dtype=np.float32).dtype == np.dtype("float32")
    with pytest.raises(ValueError, match="float32 or float64"):
        configured.with_overrides(dtype="int64")
    path = ini(tmp_path, "precision = float16\n")
    with pytest.raises(ValueError, match="float32 or float64"):
        load_e2e_calibration_configuration(path)



def test_sensing_history_controls_load_from_ini(tmp_path):
    path = tmp_path / "calibration.ini"
    path.write_text(
        "[outputs]\n"
        "CovarianceModelPath = products/covariance/model.json\n"
        "ComaModelPath = products/coma/model.json\n"
        "SensingModelPath = products/sensing/model.json\n"
        "ReconstructorModelPath = products/reconstructor/model.json\n"
        "[sensing]\nNoiseRealizations = 3\nNoiseFrames = 256\nDisplacement = 12\n"
    )
    config = load_e2e_calibration_configuration(path)
    assert config.science_psf is None
    assert config.ngs_psf is None
    assert config.ngs_metrics is None
    settings = config.sensing
    assert settings.noise_realizations == 3
    assert settings.noise_frames == 256
    assert settings.displacement == 12 * u.mas


@pytest.mark.parametrize("reference", [
    {"payload": "arrays.npz"},
    {"payload_sha256": "0" * 64},
    {"payload": "arrays.npz", "payload_sha256": "0" * 64},
])
def test_model_files_reject_former_array_file_keys(tmp_path, reference):
    path = tmp_path / "model.json"
    path.write_text(json.dumps(reference))
    with pytest.raises(ValueError, match="unsupported array file fields"):
        read_calibration_model_files(path)


@pytest.mark.parametrize("reference", [
    {"array_file": "arrays.npz"},
    {"array_file_sha256": "0" * 64},
])
def test_model_files_require_both_array_file_fields(tmp_path, reference):
    path = tmp_path / "model.json"
    path.write_text(json.dumps(reference))
    with pytest.raises(ValueError, match="must appear together"):
        read_calibration_model_files(path)


@pytest.mark.parametrize("target", ["manifest", "array_file"])
@pytest.mark.parametrize("mutation", ["change", "remove"])
def test_model_files_detect_changed_or_missing_files(tmp_path, target, mutation):
    import hashlib

    array_file = tmp_path / "arrays.npz"
    np.savez(array_file, values=[1.0, 2.0])
    manifest = tmp_path / "model.json"
    manifest.write_text(json.dumps({
        "array_file": array_file.name,
        "array_file_sha256": hashlib.sha256(array_file.read_bytes()).hexdigest(),
    }))
    model_files = read_calibration_model_files(manifest)
    path = manifest if target == "manifest" else array_file
    if mutation == "remove":
        path.unlink()
        with pytest.raises(FileNotFoundError):
            model_files.verify()
    else:
        path.write_bytes(path.read_bytes() + b"\n")
        with pytest.raises(ValueError, match="content changed"):
            model_files.verify()
