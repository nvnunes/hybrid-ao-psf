from dataclasses import FrozenInstanceError, replace

import numpy as np
import pytest
from ao_stats import PsfMetadata
from astropy import units as u

from hybrid_ao_psf import (
    DiagnosticsLevel,
    HybridDiagnostics,
    HybridRequest,
    HybridResult,
    NgsHoMetricPrediction,
    NonPsdPolicy,
    ScienceHoPsfPrediction,
)


def _metadata() -> PsfMetadata:
    return PsfMetadata(
        wavelength=1.65 * u.um,
        pixel_scale=8.8 * u.mas,
        tel_diameter=8.0 * u.m,
        tel_pupil=np.ones((4, 4)) * u.dimensionless_unscaled,
    )


def make_result(*, diagnostics: bool = True) -> HybridResult:
    psfs = np.full((2, 4, 4), 1.0 / 16.0, dtype=np.float32)
    ctot_wavefront = np.array([[[4.0, 1.0], [1.0, 3.0]]] * 2) * u.nm**2
    ctot_angle = np.array([[[1.0, 0.25], [0.25, 0.75]]] * 2) * u.mas**2
    return HybridResult(
        psfs=psfs,
        metadata=_metadata(),
        source_metadata={"airmass": 1.2 * u.dimensionless_unscaled},
        ngs_metrics=NgsHoMetricPrediction(
            ee=np.array([0.4, 0.5]) * u.dimensionless_unscaled,
            fwhm=np.array([100.0, 110.0]) * u.mas,
            sr=np.array([0.1, 0.2]) * u.dimensionless_unscaled,
        ),
        ctot_wavefront=ctot_wavefront,
        ctot_angle=ctot_angle,
        angle_to_wavefront_scale=2.0 * u.nm / u.mas,
        ngs_flux=np.array([1000.0, 2000.0]) * u.photon / u.s,
        ngs_frame_rate=np.array([500.0, 500.0]) * u.Hz,
        jitter=np.full(2, np.sqrt(1.75)) * u.mas,
        effective_mastsel_ini="[telescope]\nZenithAngle = 30\n",
        psd_clipped=np.array([False, True]),
        provenance={"science_psf_artifact": "science.pkl"},
        diagnostics=(
            HybridDiagnostics(
                psd_valid_mask=np.array([True, True]),
                ctot_angle_trace_min=1.75 * u.mas**2,
                ctot_angle_trace_max=1.75 * u.mas**2,
                ctot_angle_trace_mean=1.75 * u.mas**2,
                ctot_angle_determinant_min=0.6875 * u.mas**4,
                ctot_angle_determinant_max=0.6875 * u.mas**4,
                ctot_angle_determinant_mean=0.6875 * u.mas**4,
                pre_blur_psfs=psfs,
            )
            if diagnostics
            else None
        ),
    )


def test_request_normalizes_units_and_owns_readonly_arrays() -> None:
    science_x = np.array([0.0, 1.0])
    request = HybridRequest(
        science_x=science_x * u.arcsec,
        science_y=np.array([0.0, -1.0]) * u.arcsec,
        ngs_x=np.array([10.0]) * u.arcsec,
        ngs_y=np.array([20.0]) * u.arcsec,
        wavelength=1.65 * u.um,
        zenith_angle=30.0 * u.deg,
        ngs_flux=np.array([1000.0]) * u.photon / u.s,
        ngs_frame_rate=np.array([500.0]) * u.Hz,
        mastsel_ini="[telescope]\nZenithAngle = 30\n",
        non_psd_policy="clip",
        diagnostics_level="none",
    )
    science_x[0] = 99.0

    assert request.science_x.to_value(u.arcsec).tolist() == [0.0, 1.0]
    assert not request.science_x.value.flags.writeable
    assert request.non_psd_policy is NonPsdPolicy.CLIP
    assert request.diagnostics_level is DiagnosticsLevel.NONE
    with pytest.raises(ValueError):
        request.science_x.value[0] = 2.0
    with pytest.raises(FrozenInstanceError):
        request.wavelength = 2.2 * u.um


def test_request_rejects_mismatched_ngs_vectors() -> None:
    with pytest.raises(ValueError, match="identical shapes"):
        HybridRequest(
            science_x=np.array([0.0]) * u.arcsec,
            science_y=np.array([0.0]) * u.arcsec,
            ngs_x=np.array([1.0, 2.0]) * u.arcsec,
            ngs_y=np.array([1.0]) * u.arcsec,
            wavelength=1.65 * u.um,
            zenith_angle=30.0 * u.deg,
            ngs_flux=np.array([1000.0]) * u.photon / u.s,
            ngs_frame_rate=np.array([500.0]) * u.Hz,
            mastsel_ini="[telescope]\nZenithAngle = 30\n",
        )


def test_provider_prediction_owns_inputs() -> None:
    psfs = np.ones((1, 4, 4), dtype=np.float64)
    prediction = ScienceHoPsfPrediction(
        psfs=psfs, pixel_scale=8.8 * u.mas, metadata=_metadata()
    )
    psfs.fill(0.0)

    assert prediction.psfs.dtype == np.float32
    assert np.all(prediction.psfs == 1.0)
    assert not prediction.psfs.flags.writeable


def test_result_validates_diagnostic_shape() -> None:
    with pytest.raises(ValueError, match="must match psfs shape"):
        result = make_result(diagnostics=False)
        HybridResult(
            **{
                **result.__dict__,
                "diagnostics": HybridDiagnostics(
                    psd_valid_mask=np.array([True, True]),
                    ctot_angle_trace_min=1.75 * u.mas**2,
                    ctot_angle_trace_max=1.75 * u.mas**2,
                    ctot_angle_trace_mean=1.75 * u.mas**2,
                    ctot_angle_determinant_min=0.6875 * u.mas**4,
                    ctot_angle_determinant_max=0.6875 * u.mas**4,
                    ctot_angle_determinant_mean=0.6875 * u.mas**4,
                    pre_blur_psfs=np.ones((1, 4, 4), dtype=np.float32)
                ),
            }
        )


def test_result_rejects_inconsistent_ctot_units() -> None:
    result = make_result()

    with pytest.raises(ValueError, match="ctot_angle must equal"):
        replace(result, ctot_angle=result.ctot_angle * 2.0)


def test_result_rejects_materially_non_psd_ctot() -> None:
    result = make_result()
    invalid_wavefront = np.array([[[-1.0, 0.0], [0.0, 1.0]]] * 2) * u.nm**2

    with pytest.raises(ValueError, match="positive semidefinite"):
        replace(
            result,
            ctot_wavefront=invalid_wavefront,
            ctot_angle=invalid_wavefront.value / 4.0 * u.mas**2,
        )


def test_result_rejects_jitter_inconsistent_with_ctot() -> None:
    result = make_result()

    with pytest.raises(ValueError, match="jitter must equal"):
        replace(result, jitter=np.array([999.0, 999.0]) * u.mas)


def test_result_rejects_diagnostic_mask_inconsistent_with_ctot() -> None:
    result = make_result()
    assert result.diagnostics is not None

    with pytest.raises(ValueError, match="psd_valid_mask must match"):
        replace(
            result,
            diagnostics=replace(
                result.diagnostics,
                psd_valid_mask=np.array([True, False]),
            ),
        )


def test_result_rejects_diagnostic_summary_inconsistent_with_ctot() -> None:
    result = make_result()
    assert result.diagnostics is not None

    with pytest.raises(ValueError, match="ctot_angle_trace_min must match"):
        replace(
            result,
            diagnostics=replace(
                result.diagnostics,
                ctot_angle_trace_min=2.0 * u.mas**2,
                ctot_angle_trace_max=2.0 * u.mas**2,
                ctot_angle_trace_mean=2.0 * u.mas**2,
            ),
        )


def test_request_accepts_validation_diagnostics_level() -> None:
    request = HybridRequest(
        science_x=np.array([0.0]) * u.arcsec,
        science_y=np.array([0.0]) * u.arcsec,
        ngs_x=np.array([1.0]) * u.arcsec,
        ngs_y=np.array([1.0]) * u.arcsec,
        wavelength=1.65 * u.um,
        zenith_angle=30.0 * u.deg,
        ngs_flux=np.array([1000.0]) * u.photon / u.s,
        ngs_frame_rate=np.array([500.0]) * u.Hz,
        mastsel_ini="[telescope]\nZenithAngle = 30\n",
        diagnostics_level="validation",
    )

    assert request.diagnostics_level is DiagnosticsLevel.VALIDATION


def test_diagnostics_exposes_psd_summary_properties() -> None:
    diagnostics = make_result().diagnostics

    assert diagnostics is not None
    assert diagnostics.psd_valid_count == 2
    assert diagnostics.psd_valid_fraction == 1.0
