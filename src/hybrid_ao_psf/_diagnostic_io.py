"""Shared declarations and values for saving Hybrid diagnostics through AO Core."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import fields
from typing import Any

import numpy as np
from astropy import units as u
from numpy.typing import DTypeLike

from ._e2e_prediction import (
    E2ELowOrderDiagnostics,
    E2ELowOrderPrediction,
    _check_saved_prediction,
    _NgsDiagnostics,
)
from ._lo_model import (
    _ControlSolution,
    _LowOrderCovariances,
    _ScalarFeedback,
    _TiltCommandCovariances,
)
from ._sensing_response import SensingPrediction
from ._units import quantity_from_value
from ._validation import validate_psf_dtype
from .contracts import (
    CovarianceDiagnostics,
    DiagnosticsLevel,
    HybridDiagnostics,
    LowOrderDistribution,
    LowOrderModel,
    MastselDiagnostics,
    PsfMetricPrediction,
)

_FEEDBACK_FIELDS = {
    "effective_feedback": "interpolation_feedback_matrix",
    "strengths": "interpolation_feedback_eigenvalues",
    "projectors": "interpolation_feedback_projectors",
    "symmetric_fallback": "used_symmetric_feedback_approximation",
    "approximation_relative_error": "feedback_approximation_relative_error",
    "eigenbasis_condition": "eigenbasis_condition_number",
}


def _field(dtype: str, shape: tuple = (), unit: str | None = None) -> dict[str, Any]:
    spec = {"dtype": dtype, "shape": shape}
    if unit is not None:
        spec["unit"] = unit
    return spec


def _metric_fields(dtype: str, ngs_axis: int | str) -> dict[str, dict[str, Any]]:
    return {
        "ngs_metrics/ee": _field(dtype, (ngs_axis,), "1"),
        "ngs_metrics/fwhm": _field(dtype, (ngs_axis,), "mas"),
        "ngs_metrics/sr": _field(dtype, (ngs_axis,), "1"),
    }


def _mastsel_fields(
    dtype: str, ngs_axis: int | str, debug: bool
) -> dict[str, dict[str, Any]]:
    specs = {
        **_metric_fields(dtype, ngs_axis),
        "ngs_flux": _field(dtype, (ngs_axis,), "ph / s"),
        "frame_rate": _field(dtype, (), "Hz"),
        "angle_to_wavefront_scale": _field(dtype, (), "nm / mas"),
    }
    if debug:
        specs["effective_ini"] = _field("str")
    return specs


def _calibrated_fields(dtype: str, num_ngs: int) -> dict[str, dict[str, Any]]:
    """Declare the saved E2E prediction, including the applied Gaussian mixture."""
    ngs_tilt = 2 * num_ngs
    return {
        **_metric_fields(dtype, num_ngs),
        "photons_per_frame": _field(dtype, (num_ngs,), "ph"),
        "coma_response_coefficients": _field(dtype, (num_ngs,), "mas / nm"),
        "sensing_gains": _field(dtype, (num_ngs,), "1"),
        "wfs_noise_variances": _field(dtype, (num_ngs,), "mas2"),
        "reconstructor_noise_covariance": _field(dtype, (ngs_tilt, ngs_tilt), "mas2"),
        "reconstructor": _field(dtype, (2, ngs_tilt), "1"),
        "feedback_matrix": _field(dtype, (2, 2), "1"),
        "interpolation_feedback_matrix": _field(dtype, (2, 2), "1"),
        "interpolation_feedback_eigenvalues": _field(dtype, (2,), "1"),
        "interpolation_feedback_projectors": _field(dtype, (2, 2, 2), "1"),
        "used_symmetric_feedback_approximation": _field("bool"),
        "feedback_approximation_relative_error": _field("float64", (), "1"),
        "eigenbasis_condition_number": _field("float64", (), "1"),
        **{
            name: _field(dtype, (2, 2), "mas2")
            for name in (
                "atmospheric_command_covariance",
                "ho_command_covariance",
                "atmospheric_ho_command_covariance",
                "noise_command_covariance",
            )
        },
        **{
            name: _field(dtype, ("num_sci", 2, 2), "mas2")
            for name in (
                "science_atmospheric_command_covariance",
                "science_ho_command_covariance",
                "science_tilt_covariance",
                "atmospheric_residual_covariance",
                "ho_correction",
                "noiseless_covariance",
            )
        },
        "noise_covariance_scales": _field(dtype, (2,), "1"),
        "displacement_distribution/weights": _field(dtype, (2,), "1"),
        "displacement_distribution/covariances": _field(
            dtype, ("num_sci", 2, 2, 2), "mas2"
        ),
    }


def _metric_values(metrics: PsfMetricPrediction) -> dict[str, u.Quantity]:
    return {
        "ngs_metrics/ee": metrics.ee,
        "ngs_metrics/fwhm": metrics.fwhm,
        "ngs_metrics/sr": metrics.sr,
    }


def _mastsel_values(mastsel: MastselDiagnostics) -> dict[str, Any]:
    values = {
        **_metric_values(mastsel.ngs_metrics),
        "ngs_flux": mastsel.ngs_flux,
        "frame_rate": mastsel.frame_rate,
        "angle_to_wavefront_scale": mastsel.angle_to_wavefront_scale,
    }
    if mastsel.effective_ini is not None:
        values["effective_ini"] = mastsel.effective_ini
    return values


def _calibrated_values(prediction: E2ELowOrderPrediction) -> dict[str, Any]:
    diagnostics = prediction.diagnostics
    ngs = diagnostics.ngs
    control = diagnostics.control
    covariances = diagnostics.covariances
    distribution = prediction.displacement_distribution
    scalar_feedback = control.scalar_feedback

    values = {
        **_metric_values(ngs.metrics),
        "photons_per_frame": ngs.photons_per_frame,
        "coma_response_coefficients": ngs.coma_response_coefficients,
        "sensing_gains": ngs.sensing.gain,
        "wfs_noise_variances": ngs.sensing.variance,
        "reconstructor_noise_covariance": ngs.reconstructor_noise_covariance,
        "reconstructor": quantity_from_value(control.reconstructor, u.one),
        "feedback_matrix": quantity_from_value(control.feedback, u.one),
        "atmospheric_residual_covariance": covariances.atmospheric_residual,
        "ho_correction": covariances.ho_correction,
        "noiseless_covariance": covariances.noiseless,
        "noise_command_covariance": covariances.noise_command,
        "noise_covariance_scales": quantity_from_value(
            diagnostics.noise_covariance_scales, u.one
        ),
        "displacement_distribution/weights": quantity_from_value(
            distribution.weights, u.one
        ),
        "displacement_distribution/covariances": distribution.covariances,
    }
    for attribute, name in _FEEDBACK_FIELDS.items():
        value = getattr(scalar_feedback, attribute)
        values[name] = (
            value
            if attribute == "symmetric_fallback"
            else quantity_from_value(value, u.one)
        )
    for item in fields(_TiltCommandCovariances):
        values[f"{item.name}_covariance"] = getattr(covariances.terms, item.name)
    return values


def _read_ngs_metrics(
    values: Mapping[str, Any], dtype: np.dtype
) -> PsfMetricPrediction:
    return PsfMetricPrediction(
        ee=values["ee"], fwhm=values["fwhm"], sr=values["sr"], dtype=dtype
    )


def _read_calibrated_prediction(
    values: Mapping[str, Any], dtype: np.dtype
) -> E2ELowOrderPrediction:
    mixture = values["displacement_distribution"]
    science_count = len(mixture["covariances"])
    num_ngs = len(values["ngs_metrics"]["ee"])
    for path, spec in _calibrated_fields(dtype.name, num_ngs).items():
        value = values
        for name in path.split("/"):
            value = value[name]
        shape = tuple(
            science_count if axis == "num_sci" else axis for axis in spec["shape"]
        )
        if np.shape(value) != shape:
            raise ValueError(f"prediction {path} must have shape {shape}")

    distribution = LowOrderDistribution(
        weights=mixture["weights"].to_value(u.one),
        covariances=mixture["covariances"],
        dtype=dtype,
    )
    ngs = _NgsDiagnostics(
        photons_per_frame=values["photons_per_frame"],
        metrics=_read_ngs_metrics(values["ngs_metrics"], dtype),
        coma_response_coefficients=values["coma_response_coefficients"],
        sensing=SensingPrediction(
            gain=values["sensing_gains"],
            variance=values["wfs_noise_variances"],
            dtype=dtype,
        ),
        reconstructor_noise_covariance=values["reconstructor_noise_covariance"],
    )
    feedback = {
        attribute: values[name]
        if attribute == "symmetric_fallback"
        else values[name].to_value(u.one)
        for attribute, name in _FEEDBACK_FIELDS.items()
    }
    control = _ControlSolution(
        reconstructor=values["reconstructor"].to_value(u.one),
        feedback=values["feedback_matrix"].to_value(u.one),
        scalar_feedback=_ScalarFeedback(**feedback),
    )
    terms = _TiltCommandCovariances(
        **{
            item.name: values[f"{item.name}_covariance"]
            for item in fields(_TiltCommandCovariances)
        }
    )
    covariances = _LowOrderCovariances(
        terms=terms,
        atmospheric_residual=values["atmospheric_residual_covariance"],
        ho_correction=values["ho_correction"],
        noiseless=values["noiseless_covariance"],
        noise_command=values["noise_command_covariance"],
    )
    prediction = E2ELowOrderPrediction(
        displacement_distribution=distribution,
        diagnostics=E2ELowOrderDiagnostics(
            ngs=ngs,
            control=control,
            covariances=covariances,
            noise_covariance_scales=values["noise_covariance_scales"].to_value(u.one),
        ),
    )
    _check_saved_prediction(prediction)
    return prediction


def _restore_diagnostics(
    values: Mapping[str, Any], dtype: np.dtype
) -> HybridDiagnostics:
    """Reconstruct typed scientific diagnostics from AO Core's nested values."""
    covariance = values["covariance"]
    mastsel = None
    if "mastsel" in values:
        model = dict(values["mastsel"])
        mastsel = MastselDiagnostics(
            ngs_metrics=_read_ngs_metrics(model.pop("ngs_metrics"), dtype), **model
        )
    prediction = None
    if "e2e_calibrated" in values:
        prediction = _read_calibrated_prediction(values["e2e_calibrated"], dtype)
    return HybridDiagnostics(
        covariance=CovarianceDiagnostics(**covariance),
        mastsel=mastsel,
        e2e_calibrated=prediction,
        psfs_ho=values["psfs_ho"].to_value(u.one) if "psfs_ho" in values else None,
    )


def diagnostic_fields(
    lo_model: LowOrderModel | str,
    level: DiagnosticsLevel | str,
    *,
    dtype: DTypeLike = np.float32,
    psf_shape: tuple[int, int] | None = None,
    num_ngs: int | None = None,
) -> dict[str, dict[str, Any]]:
    """Declare the diagnostics a Hybrid simulation saves for a model and level.

    Args:
        lo_model: Selected low-order implementation.
        level: Requested diagnostic level.
        dtype: Floating-point result precision.
        psf_shape: Rows and columns of a science PSF. Required for debug.
        num_ngs: Active NGS count. Required for E2E-Calibrated debug. For
            MASTSEL, omitting it uses the AO Core ``num_ngs`` slot axis.

    Returns:
        AO Core field declarations, keyed by paths relative to ``/diagnostics``.
        The shapes describe one observation. Callers saving batches align
        values on the NGS slot axis.

    Raises:
        ValueError: If the model or level is invalid or a required dimension
            is missing.
    """
    model = LowOrderModel(lo_model)
    level = DiagnosticsLevel(level)
    precision = validate_psf_dtype(dtype).name
    if level is DiagnosticsLevel.NONE:
        return {}
    specs = {
        "covariance/ctot": _field(precision, ("num_sci", 2, 2), "mas2"),
        "covariance/psd_valid_mask": _field("bool", ("num_sci",)),
    }
    debug = level is DiagnosticsLevel.DEBUG
    if model is LowOrderModel.MASTSEL:
        specs["covariance/psd_clipped"] = _field("bool", ("num_sci",))
        specs.update(
            {
                f"mastsel/{name}": spec
                for name, spec in _mastsel_fields(
                    precision, "num_ngs" if num_ngs is None else num_ngs, debug
                ).items()
            }
        )
    elif debug:
        if num_ngs is None or num_ngs <= 0:
            raise ValueError("E2E-Calibrated debug diagnostics require the NGS count.")
        specs.update(
            {
                f"e2e_calibrated/{name}": spec
                for name, spec in _calibrated_fields(precision, num_ngs).items()
            }
        )
    if debug:
        if (
            psf_shape is None
            or len(psf_shape) != 2
            or any(size <= 0 for size in psf_shape)
        ):
            raise ValueError("Debug diagnostics require a positive science PSF shape.")
        specs["psfs_ho"] = _field(precision, ("num_sci", *psf_shape), "1")
    return specs


def diagnostic_values(diagnostics: HybridDiagnostics | None) -> dict[str, Any]:
    """Map a result's retained diagnostics to the declared AO Core paths.

    Args:
        diagnostics: Diagnostics from a Hybrid result, or None for level none.

    Returns:
        Values in one observation's science and active NGS order. Floating
        values are quantities with scientific units or unit one.
    """
    if diagnostics is None:
        return {}
    covariance = diagnostics.covariance
    values = {
        "covariance/ctot": covariance.ctot,
        "covariance/psd_valid_mask": covariance.psd_valid_mask,
    }
    if covariance.psd_clipped is not None:
        values["covariance/psd_clipped"] = covariance.psd_clipped
    if diagnostics.mastsel is not None:
        values.update(
            {
                f"mastsel/{name}": value
                for name, value in _mastsel_values(diagnostics.mastsel).items()
            }
        )
    if diagnostics.e2e_calibrated is not None:
        values.update(
            {
                f"e2e_calibrated/{name}": value
                for name, value in _calibrated_values(
                    diagnostics.e2e_calibrated
                ).items()
            }
        )
    if diagnostics.psfs_ho is not None:
        values["psfs_ho"] = u.Quantity(diagnostics.psfs_ho, u.one, copy=False)
    return values
