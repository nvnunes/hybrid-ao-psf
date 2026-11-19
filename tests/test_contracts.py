from dataclasses import FrozenInstanceError, fields, replace
from inspect import Parameter, signature

import numpy as np
import pytest
from ao_core.stats import PsfMetadata
from astropy import units as u

from hybrid_ao_psf import (
    CovarianceDiagnostics,
    DiagnosticsLevel,
    HybridCtotCorrection,
    HybridDiagnostics,
    HybridMastselStatistics,
    HybridRequest,
    HybridResult,
    MastselDiagnostics,
    PsfMetricPrediction,
    PsfPrediction,
)


def _metadata() -> PsfMetadata:
    return PsfMetadata(
        wavelength=1.65 * u.um,
        pixel_scale=8.8 * u.mas,
        tel_diameter=8.0 * u.m,
        tel_pupil=np.ones((4, 4)) * u.dimensionless_unscaled,
    )


def test_correction_owns_input_files_and_validates_definition() -> None:
    metadata = {"fit": "fit.json"}
    correction = HybridCtotCorrection(lambda ctot, context: None, metadata)
    metadata["fit"] = "changed.json"
    assert correction.input_files["fit"] == "fit.json"
    with pytest.raises(TypeError):
        correction.input_files["fit"] = "other"
    with pytest.raises(FrozenInstanceError):
        correction.apply = lambda ctot, context: ctot
    with pytest.raises(TypeError, match="callable"):
        HybridCtotCorrection(None)
    with pytest.raises(TypeError, match="input_files"):
        HybridCtotCorrection(lambda ctot, context: None, {"fit": 1})


def test_mastsel_statistics_copy_and_freeze_arrays() -> None:
    css = np.eye(4) * u.nm**2
    cas = np.ones((3, 2, 4)) * u.nm**2
    r = np.ones((3, 2, 4))
    statistics = HybridMastselStatistics(css, cas, cas[0], r, css)
    css[0, 0] = 99 * u.nm**2
    r[0, 0, 0] = 99
    assert statistics.ngs_covariance[0, 0] == 1 * u.nm**2
    assert statistics.reconstructor[0, 0, 0] == 1
    for value in (
        statistics.ngs_covariance,
        statistics.science_ngs_covariance,
        statistics.centre_ngs_covariance,
        statistics.reconstructor,
        statistics.measurement_noise,
    ):
        assert not value.flags.writeable
    with pytest.raises(ValueError, match="interleaved"):
        replace(statistics, ngs_covariance=np.eye(3) * u.nm**2)
    with pytest.raises(ValueError, match="shape"):
        replace(statistics, measurement_noise=np.eye(2) * u.nm**2)


def make_result(*, diagnostics=True, debug=True) -> HybridResult:
    psfs = np.full((2, 4, 4), 1.0 / 16.0, dtype=np.float32)
    ctot = np.array([[[1.0, 0.25], [0.25, 0.75]]] * 2, dtype=np.float32) * u.mas**2
    output_diagnostics = None
    if diagnostics:
        output_diagnostics = HybridDiagnostics(
            covariance=CovarianceDiagnostics(
                ctot=ctot, psd_valid_mask=[True, True], psd_clipped=[False, True],
            ),
            mastsel=MastselDiagnostics(
                ngs_metrics=PsfMetricPrediction(
                    ee=[0.4, 0.5] * u.one,
                    fwhm=[100.0, 110.0] * u.mas,
                    sr=[0.1, 0.2] * u.one,
                ),
                frame_rate=500.0 * u.Hz,
                ngs_flux=[1000.0, 2000.0] * u.photon / u.s,
                angle_to_wavefront_scale=2.0 * u.nm / u.mas,
                effective_ini="[telescope]\nZenithAngle = 30\n" if debug else None,
            ),
            psfs_ho=psfs if debug else None,
        )
    return HybridResult(
        positions=[[0, 0], [1, 0]] * u.arcsec,
        psfs=psfs,
        jitter=np.full(2, np.sqrt(1.75)) * u.mas,
        metadata=_metadata(),
        diagnostics=output_diagnostics,
    )


def test_request_normalizes_units_and_owns_readonly_arrays() -> None:
    science_positions = np.array([[0.0, 0.0], [1000.0, -1000.0]]) * u.mas
    request = HybridRequest(
        science_positions=science_positions,
        ngs_positions=[[10.0, 20.0]] * u.arcsec,
        ngs_flux=np.array([1000.0]) * u.photon / u.s,
        wavelength=1.65 * u.um,
        zenith_angle=30.0 * u.deg,
        diagnostics_level="none",
    )
    science_positions[0, 0] = 99.0 * u.mas

    np.testing.assert_array_equal(request.science_positions.to_value(u.arcsec), [[0, 0], [1, -1]])
    assert not request.science_positions.value.flags.writeable
    assert request.diagnostics_level is DiagnosticsLevel.NONE
    with pytest.raises(ValueError):
        request.science_positions.value[0, 0] = 2.0
    with pytest.raises(FrozenInstanceError):
        request.wavelength = 2.2 * u.um


def test_request_groups_ngs_inputs_after_coordinates() -> None:
    expected = [
        "science_positions",
        "ngs_positions",
        "ngs_flux",
        "ngs_magnitude",
    ]
    assert [item.name for item in fields(HybridRequest)[: len(expected)]] == expected
    parameters = signature(HybridRequest).parameters
    assert list(parameters)[: len(expected)] == expected
    assert all(
        parameter.kind is Parameter.KEYWORD_ONLY for parameter in parameters.values()
    )


def test_request_rejects_mismatched_ngs_vectors() -> None:
    with pytest.raises(ValueError, match="one NGS flux or magnitude per row"):
        HybridRequest(
            science_positions=[[0.0, 0.0]] * u.arcsec,
            ngs_positions=[[1.0, 1.0], [2.0, 1.0]] * u.arcsec,
            ngs_flux=np.array([1000.0]) * u.photon / u.s,
            wavelength=1.65 * u.um,
            zenith_angle=30.0 * u.deg,
        )


@pytest.mark.parametrize("field", ["science_positions", "ngs_positions"])
@pytest.mark.parametrize("shape", [(2,), (0, 2), (1, 3)])
def test_request_rejects_invalid_position_shapes(field, shape) -> None:
    positions = {
        "science_positions": [[0.0, 0.0]] * u.arcsec,
        "ngs_positions": [[1.0, 1.0]] * u.arcsec,
    }
    positions[field] = np.zeros(shape) * u.arcsec
    with pytest.raises(ValueError, match=field):
        HybridRequest(
            **positions,
            ngs_flux=[1000.0] * u.photon / u.s,
            wavelength=1.65 * u.um,
            zenith_angle=30.0 * u.deg,
        )


def test_request_requires_one_complete_ngs_input_path() -> None:
    base = {
        "science_positions": [[0.0, 0.0]] * u.arcsec,
        "ngs_positions": [[1.0, 1.0]] * u.arcsec,
        "wavelength": 1.65 * u.um,
        "zenith_angle": 30.0 * u.deg,
    }
    flux = np.array([1000.0]) * u.photon / u.s
    magnitude = np.array([12.0]) * u.mag
    with pytest.raises(ValueError, match="exactly one"):
        HybridRequest(**base)
    with pytest.raises(ValueError, match="exactly one"):
        HybridRequest(
            **base,
            ngs_flux=flux,
            ngs_magnitude=magnitude,
        )
    assert HybridRequest(**base, ngs_magnitude=magnitude).ngs_magnitude == magnitude
    with pytest.raises(ValueError, match="one NGS flux or magnitude per row"):
        HybridRequest(
            **base,
            ngs_magnitude=np.array([12.0, 13.0]) * u.mag,
        )


def test_provider_prediction_owns_inputs() -> None:
    psfs = np.ones((1, 4, 4), dtype=np.float64)
    prediction = PsfPrediction(psfs=psfs, pixel_scale=8.8 * u.mas, metadata=_metadata())
    psfs.fill(0.0)

    assert prediction.psfs.dtype == np.float32
    assert np.all(prediction.psfs == 1.0)
    assert not prediction.psfs.flags.writeable


def test_result_contains_the_science_outputs_and_physical_metadata():
    result = make_result(diagnostics=False)
    assert [item.name for item in fields(HybridResult) if item.init] == [
        "positions",
        "psfs",
        "jitter",
        "metadata",
        "diagnostics",
    ]
    np.testing.assert_array_equal(result.positions.to_value(u.arcsec), [[0, 0], [1, 0]])
    assert result.diagnostics is None
    with pytest.raises(TypeError, match="positions"):
        replace(result, positions=None)
    with pytest.raises(ValueError, match="positions must have shape"):
        replace(result, positions=[[0, 0]] * u.arcsec)


def test_result_validates_diagnostic_shape() -> None:
    result = make_result()
    diagnostics = replace(
        result.diagnostics, psfs_ho=np.ones((2, 3, 4), dtype=np.float32)
    )
    with pytest.raises(ValueError, match="must match psfs shape"):
        replace(result, diagnostics=diagnostics)


def test_mastsel_diagnostics_require_a_scalar_frame_rate() -> None:
    with pytest.raises(ValueError, match="frame_rate.*0D"):
        replace(make_result().diagnostics.mastsel, frame_rate=[500.0, 500.0] * u.Hz)


def test_request_accepts_validation_diagnostics_level() -> None:
    request = HybridRequest(
        science_positions=[[0.0, 0.0]] * u.arcsec,
        ngs_positions=[[1.0, 1.0]] * u.arcsec,
        ngs_flux=np.array([1000.0]) * u.photon / u.s,
        wavelength=1.65 * u.um,
        zenith_angle=30.0 * u.deg,
        diagnostics_level="validation",
    )

    assert request.diagnostics_level is DiagnosticsLevel.VALIDATION


def test_diagnostics_exposes_psd_summary_properties() -> None:
    diagnostics = make_result().diagnostics

    assert diagnostics is not None
    assert diagnostics.covariance.psd_valid_count == 2
    assert diagnostics.covariance.psd_valid_fraction == 1.0


@pytest.mark.parametrize("dtype", [np.float32, np.float64, None])
def test_ngs_metrics_own_immutable_values_and_unit_metadata(dtype):
    source = {
        "ee": np.array([0.4, 0.5], dtype=np.float64) * u.one,
        "fwhm": np.array([100, 110], dtype=np.float32) * u.mas,
        "sr": np.array([0.1, 0.2], dtype=np.float64) * u.one,
    }
    metrics = PsfMetricPrediction(**source, dtype=dtype)
    for name, supplied in source.items():
        expected = supplied.astype(dtype) if dtype is not None else supplied.copy()
        supplied[0] = 0 * supplied.unit
        value = getattr(metrics, name)
        np.testing.assert_array_equal(value, expected)
        assert value.dtype == expected.dtype
        # Re-enabling writes on a backing array must not unlock the result.
        backing = value
        while isinstance(backing, np.ndarray):
            with pytest.raises(ValueError):
                backing.flags.writeable = True
            if isinstance(backing, u.Quantity):
                backing *= u.s
                assert getattr(metrics, name).unit == expected.unit
            backing = backing.base
        value *= u.s
        assert getattr(metrics, name).unit == expected.unit
        np.testing.assert_array_equal(getattr(metrics, name), expected)
