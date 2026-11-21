"""Peer-model execution, independent mixture convolution and offline results."""

import subprocess
import sys
from dataclasses import replace

import h5py
import numpy as np
import pytest
from ao_core.stats import PsfMetadata
from astropy import units as u

from hybrid_ao_psf import (
    HybridRequest,
    LowOrderDistribution,
    LowOrderModel,
    MastselModel,
    PsfPrediction,
    PsfSamples,
    build_psf_interpolator,
    diagnostic_values,
    load_configuration,
    load_result,
    run_configuration,
    save_coma_calibration,
    save_covariance_calibration,
    save_psf_interpolator,
    save_psf_metric_interpolator,
    save_reconstructor_calibration,
    save_sensing_calibration,
    simulate,
    simulate_configuration,
)

from .test_e2e_model import synthetic as _synthetic
from .test_result_io import save_simulation

synthetic = _synthetic


class Science:
    def __init__(self, dtype):
        y, x = np.mgrid[-5:6, -6:7]
        self.image = np.asarray(
            np.exp(-((x - 0.5) ** 2 + 1.3 * (y + 0.4) ** 2) / 4), dtype=dtype
        )
        self.image *= dtype(0.7) / self.image.sum()
        self.dtype = dtype

    def evaluate(self, *, x, **kwargs):
        return PsfPrediction(
            np.repeat(self.image[None], len(x), axis=0),
            1 * u.mas,
            PsfMetadata(
                wavelength=1.65 * u.um,
                pixel_scale=1 * u.mas,
                tel_diameter=8 * u.m,
                tel_pupil=np.ones((4, 4)) * u.one,
            ),
            dtype=self.dtype,
        )


def request(record, indices, dtype=np.float64, diagnostics="debug"):
    return HybridRequest(
        science_positions=record.positions[[4, 5]],
        ngs_positions=record.positions[indices],
        ngs_magnitude=np.full(len(indices), 9) * u.mag,
        wavelength=1.65 * u.um,
        zenith_angle=20 * u.deg,
        dtype=dtype,
        diagnostics_level=diagnostics,
    )


def independent_blur(images, weights, covariances, pixel):
    y, x = images.shape[-2:]
    fx, fy = np.meshgrid(np.fft.fftfreq(2 * x), np.fft.fftfreq(2 * y))
    result = []
    for image, components in zip(images, covariances, strict=True):
        otf = np.zeros((2 * y, 2 * x))
        for weight, c in zip(weights, components, strict=True):
            otf += weight * np.exp(
                -2
                * np.pi**2
                * (c[0, 0] * fx**2 + 2 * c[0, 1] * fx * fy + c[1, 1] * fy**2)
                / pixel**2
            )
        padded = np.pad(
            image.astype(np.float64), ((y // 2, y - y // 2), (x // 2, x - x // 2))
        )
        result.append(
            np.maximum(
                np.fft.ifft2(np.fft.fft2(padded) * otf).real[
                    y // 2 : y // 2 + y, x // 2 : x // 2 + x
                ],
                0,
            )
        )
    return np.array(result)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("indices", [[0], [0, 8], [0, 8, 2]])
def test_calibrated_engine_matches_independent_mixture_and_roundtrip(
    synthetic, dtype, indices, tmp_path, monkeypatch
):
    model, record = synthetic
    source = Science(np.float64)

    def forbidden(*args, **kwargs):
        raise AssertionError("MASTSEL forbidden")

    monkeypatch.setattr("hybrid_ao_psf.engine._open_mastsel", forbidden)
    path = tmp_path / "result.h5"
    result = save_simulation(path, request(record, indices, dtype), source, lo_model=model)
    prediction = result.diagnostics.e2e_calibrated
    np.testing.assert_array_equal(
        prediction.displacement_distribution.covariance,
        result.diagnostics.covariance.ctot,
    )
    assert not hasattr(result, "lo_distribution")
    assert result.diagnostics.psfs_ho.dtype == dtype
    np.testing.assert_array_equal(
        result.diagnostics.psfs_ho,
        np.repeat(source.image.astype(dtype)[None], len(result.psfs), axis=0),
    )
    expected = independent_blur(
        result.diagnostics.psfs_ho,
        prediction.displacement_distribution.weights,
        prediction.displacement_distribution.covariances.value,
        1.0,
    )
    np.testing.assert_allclose(
        result.psfs,
        expected,
        rtol=2e-6 if dtype == np.float32 else 2e-13,
        atol=1e-9 if dtype == np.float32 else 1e-15,
    )
    assert result.diagnostics.mastsel is None
    assert result.diagnostics.covariance.psd_clipped is None
    assert result.psfs.dtype == dtype
    assert np.all(result.psfs.sum((1, 2)) < source.image.sum())
    np.testing.assert_array_equal(
        source.image, source.evaluate(x=[0] * u.arcsec).psfs[0]
    )
    with h5py.File(path) as root:
        assert "diagnostics/mastsel" not in root
        assert set(root["diagnostics"]) == {"covariance", "e2e_calibrated", "psfs_ho"}
        stored_prediction = root["diagnostics/e2e_calibrated"]
        assert "_total" not in stored_prediction
        assert set(stored_prediction["displacement_distribution"]) == {"weights", "covariances"}
        assert stored_prediction["sensing_gains"].attrs["units"] == "1"
    loaded = load_result(path)
    np.testing.assert_array_equal(loaded.psfs, result.psfs)
    expected_diagnostics = diagnostic_values(result.diagnostics)
    read_diagnostics = diagnostic_values(loaded.diagnostics)
    assert expected_diagnostics.keys() == read_diagnostics.keys()
    for path, value in expected_diagnostics.items():
        np.testing.assert_equal(read_diagnostics[path], value, err_msg=path)
    np.testing.assert_allclose(
        loaded.positions.to_value(u.arcsec), result.positions.to_value(u.arcsec),
        rtol=0, atol=4 * np.finfo(dtype).eps * np.max(np.abs(result.positions.value)),
    )
    assert result.diagnostics.e2e_calibrated.displacement_distribution.dtype == dtype
    # The calibrated law is observable, rather than just a second route to Ctot.
    gaussian = independent_blur(
        result.diagnostics.psfs_ho, [1], result.diagnostics.covariance.ctot.value[:, None], 1.0
    )
    assert np.linalg.norm(result.psfs - gaussian) / np.linalg.norm(result.psfs) > 1e-6


def test_e2e_validation_reports_covariance_and_psd_without_debug(synthetic, tmp_path):
    model, record = synthetic
    path = tmp_path / "validation.h5"
    save_simulation(
        path, request(record, [0], diagnostics="validation"),
        Science(np.float64), lo_model=model,
    )
    with h5py.File(path) as root:
        assert set(root["diagnostics"]) == {"covariance"}
    result = load_result(path)
    diagnostics = result.diagnostics
    assert diagnostics.covariance.ctot.shape == (len(result.positions), 2, 2)
    assert diagnostics.covariance.psd_clipped is None
    assert diagnostics.mastsel is None
    assert diagnostics.e2e_calibrated is None
    assert diagnostics.psfs_ho is None


def test_model_selection_is_explicit(synthetic):
    model, record = synthetic
    source = Science(np.float64)
    req = request(record, [0])
    mastsel = MastselModel(mastsel_ini="[telescope]", ngs_metrics=model.ngs_metrics)

    with pytest.raises(TypeError, match="lo_model"):
        simulate(req, source)
    with pytest.raises(TypeError, match="lo_model"):
        simulate(req, source, lo_model=LowOrderModel.E2E_CALIBRATED)
    with pytest.raises(ValueError, match="requires ngs_magnitude"):
        simulate(
            replace(req, ngs_magnitude=None, ngs_flux=[1] * u.photon / u.s),
            source,
            lo_model=model,
        )
    with pytest.raises(ValueError, match="require ngs_magnitude_zeropoint"):
        simulate(req, source, lo_model=mastsel)


@pytest.mark.parametrize(
    "corrupt", ["weights", "mixture", "gain", "missing", "precision"]
)
def test_result_rejects_corrupted_calibrated_state(synthetic, tmp_path, corrupt):
    model, record = synthetic
    path = tmp_path / "result.h5"
    save_simulation(path, request(record, [0]), Science(np.float64), lo_model=model)
    with h5py.File(path, "r+") as root:
        state = root["diagnostics/e2e_calibrated"]
        if corrupt == "weights":
            state["displacement_distribution/weights"][...] = [0.4, 0.6]
        if corrupt == "mixture":
            state["displacement_distribution/covariances"][...] *= 2
        if corrupt == "gain":
            state["sensing_gains"][...] = [-1]
        if corrupt == "missing":
            del state["reconstructor"]
        if corrupt == "precision":
            root["simulation/precision"][()] = "float32"
    with pytest.raises((ValueError, TypeError)):
        load_result(path)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize(
    "corrupt,error",
    [
        ("feedback_matrix", "feedback_matrix must equal"),
        ("interpolation_feedback_matrix", "effective feedback"),
        ("interpolation_feedback_projectors", "scalar decomposition"),
        ("interpolation_feedback_eigenvalues", "scalar decomposition"),
        (
            "reconstructor_noise_covariance_negative",
            "reconstructor_noise_covariance must be symmetric positive semidefinite",
        ),
        (
            "atmospheric_command_covariance_negative",
            "atmospheric_command_covariance must be symmetric positive semidefinite",
        ),
        (
            "ho_command_covariance_negative",
            "ho_command_covariance must be symmetric positive semidefinite",
        ),
        ("science_atmospheric_command_covariance", "atmospheric residual covariance disagrees"),
        ("science_ho_command_covariance", "HO covariance contribution disagrees"),
        (
            "reconstructor_noise_covariance_asymmetric",
            "reconstructor_noise_covariance must be symmetric positive semidefinite",
        ),
    ],
)
def test_result_rejects_inconsistent_scientific_diagnostics(
    synthetic, tmp_path, dtype, corrupt, error
):
    model, record = synthetic
    path = tmp_path / "diagnostics.h5"
    save_simulation(path, request(record, [0, 8], dtype), Science(dtype), lo_model=model)
    with h5py.File(path, "r+") as root:
        state = root["diagnostics/e2e_calibrated"]
        if corrupt in ("feedback_matrix", "interpolation_feedback_projectors"):
            state[corrupt][...] = 0
        elif corrupt in ("interpolation_feedback_matrix", "interpolation_feedback_eigenvalues"):
            state[corrupt][...] *= 2
        elif corrupt == "reconstructor_noise_covariance_negative":
            state["reconstructor_noise_covariance"][...] = -0.1 * np.eye(4)
        elif corrupt == "reconstructor_noise_covariance_asymmetric":
            state["reconstructor_noise_covariance"][0, 1] += 0.1
        elif corrupt.endswith("_negative"):
            state[corrupt.removesuffix("_negative")][...] = -100 * np.eye(2)
        elif corrupt.startswith("science_"):
            state[corrupt][0, 0, 0] += 0.1
    with pytest.raises(ValueError, match=error):
        load_result(path)


def test_result_rejects_saved_calibrated_clipping(synthetic, tmp_path):
    path = tmp_path / "result.h5"
    model, record = synthetic
    save_simulation(path, request(record, [0]), Science(np.float64), lo_model=model)
    with h5py.File(path, "r+") as root:
        root["diagnostics/covariance"].create_dataset("psd_clipped", data=[True, True])
        specs = root["simulation/diagnostic_fields/covariance"].create_group("psd_clipped")
        specs.create_dataset("dtype", data="bool")
        specs.create_dataset("shape", data=["num_sci"], dtype=h5py.string_dtype())

    with pytest.raises(ValueError, match="do not apply covariance clipping"):
        load_result(path)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_result_rejects_nonprojectors_at_repeated_strengths(synthetic, tmp_path, dtype):
    model, record = synthetic
    path = tmp_path / "nonprojectors.h5"
    result = save_simulation(path, request(record, [0], dtype), Science(dtype), lo_model=model)
    eigenvalues = result.diagnostics.e2e_calibrated.diagnostics.control.scalar_feedback.strengths
    np.testing.assert_array_equal(eigenvalues, np.repeat(eigenvalues[0], 2))
    with h5py.File(path, "r+") as root:
        root["diagnostics/e2e_calibrated/interpolation_feedback_projectors"][...] = [2 * np.eye(2), -np.eye(2)]
    with pytest.raises(ValueError, match="idempotent projectors"):
        load_result(path)


@pytest.fixture
def configured(synthetic, tmp_path):
    model, record = synthetic
    for name, save in (
        ("covariance", save_covariance_calibration),
        ("coma", save_coma_calibration),
        ("sensing", save_sensing_calibration),
        ("reconstructor", save_reconstructor_calibration),
    ):
        save(getattr(model, name), tmp_path / name / "model.json")
    save_psf_metric_interpolator(model.ngs_metrics, tmp_path / "metrics.pkl")
    provider = Science(np.float64)
    image = provider.image
    xy = record.positions.value
    save_psf_interpolator(
        build_psf_interpolator(
            PsfSamples(
                x=xy[:, 0] * u.arcsec,
                y=xy[:, 1] * u.arcsec,
                psfs=np.repeat(image[None], len(xy), axis=0),
                wavelength=1.65 * u.um,
                pixel_scale=1 * u.mas,
                tel_diameter=8 * u.m,
                tel_pupil=np.ones((4, 4)) * u.one,
                zenith_angle=20 * u.deg,
            ),
            dtype=np.float64,
        ),
        tmp_path / "science.pkl",
    )
    ini = tmp_path / "prediction.ini"
    ini.write_text(
        "[settings]\nPrecision = float64\n"
        "[inputs]\n"
        "CovarianceModelPath = covariance/model.json\n"
        "ComaModelPath = coma/model.json\n"
        "SensingModelPath = sensing/model.json\n"
        "ReconstructorModelPath = reconstructor/model.json\n"
        "SciencePSFPath = science.pkl\nNGSMetricsPath = metrics.pkl\n"
        "[outputs]\nOutputPath = result.h5\n"
        "[native]\nsource = unavailable\n"
        "[telescope]\nZenithAngle = 20\n"
        "[sources_science]\nWavelength = [1.65e-6]\n"
        "[hybrid]\nLOModel = e2e-calibrated\nScienceX = [0,0]\nScienceY = [0,1]\n"
        "NGSX = [-1,1]\nNGSY = [-1,1]\nNGSMagnitude = [9,9]\n"
    )
    return ini


def test_standalone_and_cli_are_native_free_and_analysis_is_self_contained(
    configured, tmp_path
):
    config = load_configuration(configured)
    assert config.request.dtype == np.float64 and config.mastsel_ini is None
    np.testing.assert_array_equal(
        config.request.science_positions.to_value(u.arcsec), [[0, 0], [0, 1]]
    )
    np.testing.assert_array_equal(
        config.request.ngs_positions.to_value(u.arcsec), [[-1, -1], [1, 1]]
    )
    assert load_configuration(configured, dtype=np.float32).request.dtype == np.float32
    memory_result = simulate_configuration(config)
    assert not config.output.exists()
    direct = run_configuration(configured)
    np.testing.assert_array_equal(memory_result.psfs, direct.psfs)
    saved = load_result(config.output)
    np.testing.assert_array_equal(direct.jitter, saved.jitter)
    expected_files = {
        "science_ho_psf_interpolator_path": str(tmp_path / "science.pkl"),
        "ngs_ho_metric_interpolator_path": str(tmp_path / "metrics.pkl"),
        **{
            f"{family}_model_path": str(tmp_path / family / "model.json")
            for family in ("covariance", "coma", "sensing", "reconstructor")
        },
    }
    with h5py.File(config.output) as output:
        assert output["simulation/name"].asstr()[()] == "e2e-calibrated"
        assert output["simulation/base_config"].asstr()[()] == configured.read_text()
        assert "diagnostics" not in output
        assert {
            key: output["simulation"][key].asstr()[()] for key in expected_files
        } == expected_files
        assert output["options/zenith_angle"][()] == 20.0
        np.testing.assert_allclose(output["options/ngs_r"], [np.sqrt(2), np.sqrt(2)])
        np.testing.assert_array_equal(output["options/ngs_theta"], [225.0, 45.0])
        np.testing.assert_array_equal(output["options/ngs_magnitude"], [9.0, 9.0])
        assert "ngs_flux" not in output["options"]
    code = """
import builtins,sys
original=builtins.__import__
def guarded(name,*args,**kwargs):
    if name.startswith(('mastsel','girmos_oomao','ao_tools','oomao')): raise AssertionError(name)
    return original(name,*args,**kwargs)
builtins.__import__=guarded
from hybrid_ao_psf.cli import main
main([sys.argv[1],'--output',sys.argv[2]])
"""
    subprocess.run(
        [sys.executable, "-c", code, str(configured), str(tmp_path / "cli.h5")],
        check=True,
        capture_output=True,
        text=True,
    )
    loaded = load_result(tmp_path / "cli.h5")
    np.testing.assert_array_equal(direct.psfs, loaded.psfs)
    for path in tmp_path.rglob("model.json"):
        path.unlink()
    (tmp_path / "science.pkl").unlink()
    (tmp_path / "metrics.pkl").unlink()
    np.testing.assert_array_equal(
        load_result(tmp_path / "cli.h5").psfs,
        loaded.psfs,
    )


def test_distribution_owns_values_and_units():
    law = LowOrderDistribution(
        weights=[0.7, 0.3], covariances=np.ones((1, 2, 2, 2)) * u.mas**2
    )
    for q in (law.covariances, law.weights):
        value = q
        while isinstance(value, np.ndarray):
            with pytest.raises(ValueError):
                value.flags.writeable = True
            if isinstance(value, u.Quantity):
                value *= u.s
                assert law.covariances.unit == u.mas**2
            value = value.base


@pytest.mark.parametrize(
    "field", ["precision", "ngs_flux_photon_per_s", "ngs_metric_file"]
)
def test_e2e_configuration_rejects_foreign_hybrid_fields(configured, field):
    with configured.open("a") as stream:
        stream.write(f"{field} = ignored\n")
    with pytest.raises(ValueError, match="unsupported E2E"):
        load_configuration(configured)


@pytest.mark.parametrize("key", ["SciencePSFPath", "NGSMetricsPath", "CovarianceModelPath"])
def test_e2e_prediction_requires_its_input_paths(configured, key):
    lines = configured.read_text().splitlines(keepends=True)
    configured.write_text("".join(line for line in lines if not line.startswith(f"{key} = ")))
    with pytest.raises(ValueError, match=key):
        load_configuration(configured)


def test_engine_result_owned_backing_and_unit_metadata(synthetic):
    model, record = synthetic
    result = simulate(request(record, [0]), Science(np.float64), lo_model=model)
    for owner, name in (
        (result, "psfs"),
        (result.diagnostics.covariance, "ctot"),
        (result, "positions"),
        (result.diagnostics, "psfs_ho"),
    ):
        original = getattr(owner, name)
        expected = original.copy()
        value = original
        while isinstance(value, np.ndarray):
            with pytest.raises(ValueError):
                value.flags.writeable = True
            if isinstance(value, u.Quantity):
                value *= u.s
                assert getattr(owner, name).unit == expected.unit
            value = value.base
        np.testing.assert_array_equal(getattr(owner, name), expected)
