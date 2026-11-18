"""Scientific result and validation for calibrated low-order prediction."""

from __future__ import annotations

from dataclasses import dataclass, fields, replace

import numpy as np
from astropy import units as u
from numpy.typing import DTypeLike, NDArray

from ._calibration_contracts import _quantity
from ._lo_model import (
    _feedback_matrix,
    _TiltCommandCovariances,
    _validate_scalar_decomposition,
)
from ._units import _PhysicalRecord
from ._validation import validate_psf_dtype
from .contracts import PsfMetricPrediction


def _immutable_array(value: object, dtype: np.dtype) -> NDArray:
    value = np.ascontiguousarray(value, dtype=dtype)
    if not np.isfinite(value).all():
        raise ValueError("calibrated prediction contains nonfinite values")
    return np.frombuffer(value.tobytes(), dtype=value.dtype).reshape(value.shape)


@dataclass(frozen=True, init=False, eq=False)
class E2ELowOrderPrediction(_PhysicalRecord):
    """Immutable scientific result from E2ECalibratedModel.predict.

    All matrices use physical x/y axes. Angular covariances are in mas squared.
    Results are created by the model, not by a caller-supplied constructor.

    Attributes:
        dtype: Selected runtime float32 or float64 arithmetic.
        photons_per_frame: Incident per-NGS photon counts, (N,).
        metrics: Prepared NGS EE, FWHM and Strehl vectors.
        coma_response: Per-NGS h, (N,), in mas per nm OPD.
        gains: Per-NGS dimensionless g, (N,).
        noise_variances: Per-NGS per-axis v, (N,), in mas squared.
        noise_prior: Separate pre-filter reconstructor prior, (2N,2N).
        reconstructor: Absolute field-centre LO weights, (2,2N).
        feedback: Actual B=sum(g_g R_g), (2,2).
        effective_feedback: B used for the scalar covariance interpolators, (2,2).
        strengths: Two real eigenvalues of effective_feedback.
        projectors: Corresponding, generally non-orthogonal (2,2,2) projections.
        symmetric_feedback_fallback: Whether the complex-eigenvalue symmetric
            approximation was needed for covariance interpolation only.
        feedback_approximation_relative_error: Relative Frobenius change in B.
        eigenbasis_condition: Condition number of the chosen eigenvector basis.
        atmospheric_command: Cov(u_atm), (2,2).
        ho_command: Cov(u_HO), (2,2).
        atmospheric_ho_command: Cov(u_atm,u_HO), (2,2); not generally symmetric.
        science_atmospheric_command: Cov(t_science,u_atm), (S,2,2).
        science_ho_command: Cov(t_science,u_HO), (S,2,2).
        uncorrected: Atmospheric science-tilt covariance, (S,2,2).
        atmospheric: Atmospheric residual covariance, (S,2,2).
        ho: HO correction including both cross terms, (S,2,2); may be indefinite.
        noiseless: Atmospheric plus HO covariance, (S,2,2).
        noise: Stationary Lyapunov noise-command covariance using actual B, (2,2).
        total: Noiseless plus noise covariance, (S,2,2).
        mixture_weights: Narrow/broad noise-mixture weights, (2,).
        noise_scales: Covariance-preserving narrow/broad scales, (2,).
        mixture_covariances: Full displacement Gaussian component covariances,
            (S,2,2,2), ordered science, mixture component, x, y.

    No per-component PSD clipping is applied. The model rejects a noiseless
    covariance with an eigenvalue below -1e-8 mas squared; tiny roundoff is
    retained explicitly rather than hidden by conditioning.
    """

    dtype: np.dtype
    photons_per_frame: u.Quantity
    metrics: PsfMetricPrediction
    coma_response: u.Quantity
    gains: u.Quantity
    noise_variances: u.Quantity
    noise_prior: u.Quantity
    reconstructor: np.ndarray
    feedback: np.ndarray
    effective_feedback: np.ndarray
    strengths: np.ndarray
    projectors: np.ndarray
    symmetric_feedback_fallback: bool
    feedback_approximation_relative_error: float
    eigenbasis_condition: float
    atmospheric_command: u.Quantity
    ho_command: u.Quantity
    atmospheric_ho_command: u.Quantity
    science_atmospheric_command: u.Quantity
    science_ho_command: u.Quantity
    uncorrected: u.Quantity
    atmospheric: u.Quantity
    ho: u.Quantity
    noiseless: u.Quantity
    noise: u.Quantity
    total: u.Quantity
    mixture_weights: np.ndarray
    noise_scales: np.ndarray
    mixture_covariances: u.Quantity

    def __init__(self):
        raise TypeError("Use E2ECalibratedModel.predict")

    @classmethod
    def _create(cls, dtype: DTypeLike, **values: object) -> E2ELowOrderPrediction:
        """Validate self-contained prediction state for runtime and I/O.

        Check feedback/reconstruction, scalar decomposition, covariance priors
        and displacement-mixture identities without modifying stored values.
        """
        dtype = validate_psf_dtype(dtype)
        expected = {item.name for item in fields(cls)} - {"dtype"}
        if set(values) != expected:
            raise ValueError(
                "prediction state must contain exactly its declared fields"
            )
        result = object.__new__(cls)
        object.__setattr__(result, "dtype", dtype)
        _prepare_prediction_fields(result, values, dtype)
        _validate_prediction_shapes(result)
        tolerance = max(1e-10, 32 * np.finfo(dtype).eps)
        _validate_feedback(result, tolerance)
        _validate_covariance_components(result, tolerance)
        _validate_displacement_mixture(result, tolerance)
        return result


def _prepare_prediction_fields(
    result: E2ELowOrderPrediction, values: dict[str, object], dtype: np.dtype
) -> None:
    """Store each prediction field with its required units and immutable backing."""
    units = {
        "photons_per_frame": u.photon,
        "coma_response": u.mas / u.nm,
        "gains": u.one,
        "noise_variances": u.mas**2,
        "noise_prior": u.mas**2,
        **{
            name: u.mas**2
            for name in (
                "atmospheric_command",
                "ho_command",
                "atmospheric_ho_command",
                "science_atmospheric_command",
                "science_ho_command",
                "uncorrected",
                "atmospheric",
                "ho",
                "noiseless",
                "noise",
                "total",
                "mixture_covariances",
            )
        },
    }
    for name, value in values.items():
        if name in units:
            if not isinstance(value, u.Quantity):
                raise TypeError(f"{name} must be a quantity")
            value = _quantity(value, units[name], name, value.ndim, dtype=dtype)
        elif name == "metrics":
            if not isinstance(value, PsfMetricPrediction):
                raise TypeError("metrics must be PsfMetricPrediction")
            value = replace(value, dtype=dtype)
        elif name == "symmetric_feedback_fallback":
            if not isinstance(value, (bool, np.bool_)):
                raise TypeError("feedback fallback must be boolean")
            value = bool(value)
        elif name in (
            "feedback_approximation_relative_error",
            "eigenbasis_condition",
        ):
            if np.ndim(value) != 0 or not np.isfinite(value) or value < 0:
                raise ValueError(f"invalid {name}")
            value = float(value)
        else:
            if np.asarray(value).dtype.kind not in "fiu":
                raise ValueError(f"{name} must be real")
            value = _immutable_array(value, dtype)
        object.__setattr__(result, name, value)


def _validate_prediction_shapes(result: E2ELowOrderPrediction) -> None:
    """Check shared NGS and science field dimensions across prediction fields."""
    n = len(result.metrics.ee)
    if result.uncorrected.ndim != 3:
        raise ValueError("uncorrected covariance must be (science,2,2)")
    count = len(result.uncorrected)
    shapes = {
        **{
            key: (n,)
            for key in (
                "photons_per_frame",
                "coma_response",
                "gains",
                "noise_variances",
            )
        },
        "noise_prior": (2 * n, 2 * n),
        "reconstructor": (2, 2 * n),
        "feedback": (2, 2),
        "effective_feedback": (2, 2),
        "strengths": (2,),
        "projectors": (2, 2, 2),
        "mixture_weights": (2,),
        "noise_scales": (2,),
        **{
            key: (2, 2)
            for key in (
                "atmospheric_command",
                "ho_command",
                "atmospheric_ho_command",
                "noise",
            )
        },
        **{
            key: (count, 2, 2)
            for key in (
                "science_atmospheric_command",
                "science_ho_command",
                "uncorrected",
                "atmospheric",
                "ho",
                "noiseless",
                "total",
            )
        },
        "mixture_covariances": (count, 2, 2, 2),
    }
    for name, shape in shapes.items():
        if getattr(result, name).shape != shape:
            raise ValueError(f"prediction {name} must have shape {shape}")
    if any(
        np.any(getattr(result, key).value <= 0)
        for key in (
            "photons_per_frame",
            "coma_response",
            "gains",
            "noise_variances",
        )
    ):
        raise ValueError(
            "positive photon counts and WFS response/noise values required"
        )


def _validate_feedback(result: E2ELowOrderPrediction, tolerance: float) -> None:
    """Verify actual feedback and the scalar decomposition used by the interpolator."""
    if not np.allclose(
        result.feedback,
        _feedback_matrix(result.reconstructor, result.gains.value),
        rtol=tolerance,
        atol=tolerance,
    ):
        raise ValueError("feedback must equal the gain-weighted reconstructor sum")
    _validate_scalar_decomposition(
        result.feedback,
        result.effective_feedback,
        result.strengths,
        result.projectors,
        result.symmetric_feedback_fallback,
    )


def _validate_covariance_components(
    result: E2ELowOrderPrediction, tolerance: float
) -> None:
    """Check the physical command and residual covariance identities."""
    for name in (
        "noise_prior",
        "atmospheric_command",
        "ho_command",
        "uncorrected",
        "atmospheric",
        "noiseless",
        "noise",
        "total",
    ):
        value = getattr(result, name).value
        if (
            not np.allclose(value, value.swapaxes(-1, -2), rtol=tolerance, atol=1e-8)
            or np.linalg.eigvalsh(value).min() < -1e-8
        ):
            raise ValueError(
                f"prediction {name} must be symmetric positive semidefinite"
            )

    terms = _TiltCommandCovariances(
        science_tilt=result.uncorrected.value,
        atmospheric_command=result.atmospheric_command.value,
        ho_command=result.ho_command.value,
        atmospheric_ho_command=result.atmospheric_ho_command.value,
        science_atmospheric_command=result.science_atmospheric_command.value,
        science_ho_command=result.science_ho_command.value,
    )
    atmospheric, ho = terms.residual_covariances()
    if not np.allclose(
        result.atmospheric.value, atmospheric, rtol=tolerance, atol=1e-8
    ):
        raise ValueError(
            "atmospheric residual covariance disagrees with stored command terms"
        )
    if not np.allclose(result.ho.value, ho, rtol=tolerance, atol=1e-8):
        raise ValueError(
            "HO covariance contribution disagrees with stored command terms"
        )
    if not np.allclose(
        result.noiseless.value,
        result.atmospheric.value + result.ho.value,
        rtol=tolerance,
        atol=1e-8,
    ):
        raise ValueError("noiseless covariance must equal atmospheric plus HO")
    if not np.allclose(
        result.total.value,
        result.noiseless.value + result.noise.value,
        rtol=tolerance,
        atol=1e-8,
    ):
        raise ValueError("total covariance must equal noiseless plus noise")


def _validate_displacement_mixture(
    result: E2ELowOrderPrediction, tolerance: float
) -> None:
    """Check that the applied Gaussian mixture preserves the total covariance."""
    if np.any(result.mixture_weights <= 0) or not np.isclose(
        result.mixture_weights.sum(), 1, rtol=tolerance
    ):
        raise ValueError("positive mixture weights must sum to one")
    if np.any(result.noise_scales <= 0) or not np.isclose(
        result.mixture_weights @ result.noise_scales, 1, rtol=tolerance
    ):
        raise ValueError("noise mixture must preserve covariance")
    expected = (
        result.noiseless.value[:, None]
        + result.noise_scales[None, :, None, None] * result.noise.value
    )
    if not np.allclose(
        result.mixture_covariances.value, expected, rtol=tolerance, atol=1e-8
    ):
        raise ValueError("mixture covariances differ from noiseless and scaled noise")

    value = result.mixture_covariances.value
    if (
        not np.allclose(value, value.swapaxes(-1, -2), rtol=tolerance, atol=1e-8)
        or np.linalg.eigvalsh(value).min() < -1e-8
    ):
        raise ValueError(
            "prediction mixture_covariances must be symmetric positive semidefinite"
        )
