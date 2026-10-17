"""Peer-model execution, independent mixture convolution and offline results."""

import subprocess
import sys
from dataclasses import fields, replace

import h5py
import numpy as np
import pytest
from ao_stats import PsfMetadata
from astropy import units as u

from hybrid_ao_psf import (
    HybridRequest,
    LowOrderDistribution,
    LowOrderModel,
    ScienceHoPsfPrediction,
    ScienceHoPsfSamples,
    build_science_ho_psf_interpolator,
    load_configuration,
    load_result,
    run_configuration,
    save_coma_calibration,
    save_covariance_calibration,
    save_ngs_ho_metric_interpolator,
    save_reconstructor_calibration,
    save_result,
    save_science_ho_psf_interpolator,
    save_sensing_calibration,
    simulate,
)

from .test_contracts import make_result
from .test_e2e_model import synthetic as _synthetic

synthetic = _synthetic


class Science:
    def __init__(self, dtype):
        y, x = np.mgrid[-5:6, -6:7]
        self.image = np.asarray(
            np.exp(-((x - 0.5) ** 2 + 1.3 * (y + 0.4) ** 2) / 4), dtype=dtype
        )
        self.image *= dtype(0.7) / self.image.sum()
        self.dtype = dtype

    def predict(self, *, x, **kwargs):
        return ScienceHoPsfPrediction(
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
        science_x=record.positions[[4, 5], 0],
        science_y=record.positions[[4, 5], 1],
        ngs_x=record.positions[indices, 0],
        ngs_y=record.positions[indices, 1],
        ngs_magnitude=np.full(len(indices), 9) * u.mag,
        wavelength=1.65 * u.um,
        zenith_angle=20 * u.deg,
        lo_model="e2e-calibrated",
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
    source = Science(dtype)

    def forbidden(*args, **kwargs):
        raise AssertionError("MASTSEL forbidden")

    monkeypatch.setattr("hybrid_ao_psf.engine._prepare_mastsel_context", forbidden)
    result = simulate(request(record, indices, dtype), source, e2e_model=model)
    expected = independent_blur(
        result.diagnostics.pre_blur_psfs,
        result.lo_distribution.weights,
        result.lo_distribution.covariances.value,
        1.0,
    )
    np.testing.assert_allclose(
        result.psfs,
        expected,
        rtol=2e-6 if dtype == np.float32 else 2e-13,
        atol=1e-9 if dtype == np.float32 else 1e-15,
    )
    assert result.lo_model is LowOrderModel.E2E_CALIBRATED
    assert (
        result.ctot_wavefront
        is result.angle_to_wavefront_scale
        is result.effective_mastsel_ini
        is result.ngs_flux
        is None
    )
    assert not result.psd_clipped.any()
    assert result.psfs.dtype == dtype
    assert np.all(result.psfs.sum((1, 2)) < source.image.sum())
    np.testing.assert_array_equal(
        source.image, source.predict(x=[0] * u.arcsec).psfs[0]
    )
    path = save_result(tmp_path / "result.h5", result)
    with h5py.File(path) as root:
        assert root.attrs["lo_model"] == "e2e-calibrated"
        assert "effective_mastsel_ini" not in root["configuration"]
        assert "wavefront" not in root["result/ctot"]
        assert "flux" not in root["result/ngs"]
    loaded = load_result(path)
    np.testing.assert_array_equal(loaded.psfs, result.psfs)
    np.testing.assert_array_equal(
        loaded.lo_distribution.covariances, result.lo_distribution.covariances
    )
    for field in fields(result.e2e_prediction):
        if field.name in ("metrics",):
            continue
        np.testing.assert_equal(
            getattr(loaded.e2e_prediction, field.name),
            getattr(result.e2e_prediction, field.name),
        )
    np.testing.assert_array_equal(loaded.science_positions, result.science_positions)
    np.testing.assert_array_equal(loaded.ngs_positions, result.ngs_positions)
    assert loaded.zenith_angle == result.zenith_angle
    assert dict(loaded.provenance) == dict(result.provenance)
    assert result.e2e_prediction.dtype == dtype
    # The calibrated law is observable, rather than just a second route to Ctot.
    gaussian = independent_blur(
        result.diagnostics.pre_blur_psfs, [1], result.ctot_angle.value[:, None], 1.0
    )
    assert np.linalg.norm(result.psfs - gaussian) / np.linalg.norm(result.psfs) > 1e-6


@pytest.mark.parametrize(
    "changes,error",
    [
        ({"mastsel_ini": "[telescope]"}, "mastsel_ini"),
        ({"ngs_magnitude_zeropoint": 1 * u.photon / (u.m**2 * u.s)}, "photometry"),
        (
            {"ngs_magnitude": None, "ngs_flux": [1] * u.photon / u.s},
            "require ngs_magnitude",
        ),
        ({"non_psd_policy": "clip"}, "non_psd_policy"),
    ],
)
def test_request_rejects_cross_model_inputs(synthetic, changes, error):
    _, record = synthetic
    with pytest.raises(ValueError, match=error):
        replace(request(record, [0]), **changes)


def test_model_selection_is_explicit(synthetic):
    model, record = synthetic
    source = Science(np.float64)
    req = request(record, [0])
    with pytest.raises(TypeError, match="requires e2e_model"):
        simulate(req, source)
    with pytest.raises(ValueError, match="NGS metrics"):
        simulate(req, source, model.ngs_metrics, e2e_model=model)
    with pytest.raises(ValueError, match="MASTSEL execution"):
        simulate(
            replace(
                req,
                lo_model="mastsel",
                ngs_magnitude=None,
                ngs_flux=[1] * u.photon / u.s,
                mastsel_ini="[telescope]",
            ),
            source,
            model.ngs_metrics,
            e2e_model=model,
        )


@pytest.mark.parametrize(
    "corrupt", ["weights", "mixture", "gain", "missing", "model", "precision"]
)
def test_result_rejects_corrupted_calibrated_state(synthetic, tmp_path, corrupt):
    model, record = synthetic
    result = simulate(request(record, [0]), Science(np.float64), e2e_model=model)
    path = save_result(tmp_path / "result.h5", result)
    with h5py.File(path, "r+") as root:
        if corrupt == "weights":
            root["result/lo_distribution/weights"][...] = [0.4, 0.6]
        if corrupt == "mixture":
            root["result/e2e_prediction/mixture_covariances"][...] *= 2
        if corrupt == "gain":
            root["result/e2e_prediction/gains"][...] = [-1]
        if corrupt == "missing":
            del root["result/e2e_prediction/reconstructor"]
        if corrupt == "model":
            del root.attrs["lo_model"]
        if corrupt == "precision":
            root.attrs["precision"] = "float32"
    with pytest.raises((ValueError, TypeError)):
        load_result(path)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize(
    "corrupt,error",
    [
        ("feedback", "feedback must equal"),
        ("effective_feedback", "effective feedback"),
        ("projectors", "scalar decomposition"),
        ("strengths", "scalar decomposition"),
        ("noise_prior_negative", "noise_prior must be symmetric positive semidefinite"),
        (
            "noise_prior_asymmetric",
            "noise_prior must be symmetric positive semidefinite",
        ),
    ],
)
def test_result_rejects_inconsistent_scientific_diagnostics(
    synthetic, tmp_path, dtype, corrupt, error
):
    model, record = synthetic
    result = simulate(request(record, [0, 8], dtype), Science(dtype), e2e_model=model)
    path = save_result(tmp_path / "diagnostics.h5", result)
    with h5py.File(path, "r+") as root:
        state = root["result/e2e_prediction"]
        if corrupt in ("feedback", "projectors"):
            state[corrupt][...] = 0
        elif corrupt in ("effective_feedback", "strengths"):
            state[corrupt][...] *= 2
        elif corrupt == "noise_prior_negative":
            state["noise_prior"][...] = -0.1 * np.eye(4)
        elif corrupt == "noise_prior_asymmetric":
            state["noise_prior"][0, 1] += 0.1
    with pytest.raises(ValueError, match=error):
        load_result(path)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_result_rejects_nonprojectors_at_repeated_strengths(synthetic, tmp_path, dtype):
    model, record = synthetic
    result = simulate(request(record, [0], dtype), Science(dtype), e2e_model=model)
    np.testing.assert_array_equal(
        result.e2e_prediction.strengths,
        np.repeat(result.e2e_prediction.strengths[0], 2),
    )
    path = save_result(tmp_path / "nonprojectors.h5", result)
    with h5py.File(path, "r+") as root:
        root["result/e2e_prediction/projectors"][...] = [2 * np.eye(2), -np.eye(2)]
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
    save_ngs_ho_metric_interpolator(model.ngs_metrics, tmp_path / "metrics.pkl")
    provider = Science(np.float64)
    image = provider.image
    xy = record.positions.value
    save_science_ho_psf_interpolator(
        build_science_ho_psf_interpolator(
            ScienceHoPsfSamples(
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
        "[e2e]\nprecision = float64\n"
        + "".join(
            f"{f}_model = {f}/model.json\n"
            for f in ("covariance", "coma", "sensing", "reconstructor")
        )
        + "science_psf = science.pkl\nngs_metrics = metrics.pkl\nngs_psf = unavailable.pkl\n[native]\nsource = unavailable\n"
        + "[hybrid]\nlo_model = e2e-calibrated\noutput = result.h5\nscience_x_arcsec = [0,0]\nscience_y_arcsec = [0,1]\nngs_x_arcsec = [-1,1]\nngs_y_arcsec = [-1,1]\nngs_magnitude = [9,9]\nwavelength_um = 1.65\nzenith_angle_deg = 20\n"
    )
    return ini


def test_standalone_and_cli_are_native_free_and_analysis_is_self_contained(
    configured, tmp_path
):
    config = load_configuration(configured)
    assert config.request.dtype == np.float64 and config.request.mastsel_ini is None
    assert load_configuration(configured, dtype=np.float32).request.dtype == np.float32
    direct = run_configuration(configured)
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
        load_result(tmp_path / "cli.h5").lo_distribution.covariances,
        loaded.lo_distribution.covariances,
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


@pytest.mark.parametrize("version", [1, 2])
def test_legacy_mastsel_loads_without_model_or_distribution(version, tmp_path):
    expected = replace(make_result(), dtype=np.float64)
    path = save_result(tmp_path / "legacy.h5", expected)
    with h5py.File(path, "r+") as root:
        root.attrs["schema_version"] = version
        del root.attrs["lo_model"]
        del root["result/lo_distribution"]
        del root["result/coordinates"]
        if version == 1:
            del root.attrs["precision"]
    loaded = load_result(path)
    assert loaded.lo_model is LowOrderModel.MASTSEL
    assert loaded.e2e_prediction is None
    assert loaded.science_positions is None
    assert loaded.dtype == (None if version == 1 else np.dtype("float64"))
    np.testing.assert_array_equal(loaded.psfs, expected.psfs)
    np.testing.assert_array_equal(loaded.ctot_wavefront, expected.ctot_wavefront)
    np.testing.assert_array_equal(
        loaded.lo_distribution.covariance, expected.ctot_angle
    )


@pytest.mark.parametrize(
    "field", ["precision", "ngs_flux_photon_per_s", "ngs_metric_artifact"]
)
def test_e2e_configuration_rejects_foreign_hybrid_fields(configured, field):
    with configured.open("a") as stream:
        stream.write(f"{field} = ignored\n")
    with pytest.raises(ValueError, match="unsupported E2E"):
        load_configuration(configured)


def test_engine_result_owned_backing_and_unit_metadata(synthetic):
    model, record = synthetic
    result = simulate(request(record, [0]), Science(np.float64), e2e_model=model)
    for owner, name in (
        (result, "psfs"),
        (result, "ctot_angle"),
        (result, "science_positions"),
        (result, "zenith_angle"),
        (result.diagnostics, "pre_blur_psfs"),
        (result.diagnostics, "ctot_angle_trace_min"),
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
    with pytest.raises(ValueError, match="do not apply covariance clipping"):
        replace(result, psd_clipped=np.ones(len(result.psfs), dtype=bool))
