"""Scientific result and validation for calibrated low-order prediction."""

from __future__ import annotations

from dataclasses import dataclass, field, fields, replace

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
from .contracts import LowOrderDistribution, PsfMetricPrediction


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
        ngs_metrics: High-order EE, FWHM and Strehl in NGS order.
        coma_response_coefficients: WFS tilt response to coma h per NGS, (N,),
            in mas per nm OPD.
        sensing_gains: Dimensionless WFS response g per NGS, (N,).
        wfs_noise_variances: WFS noise variance v per NGS per axis, (N,),
            in mas squared.
        reconstructor_noise_covariance: Pre-filter measurement-noise covariance
            used to choose the reconstructor weights, (2N,2N).
        reconstructor: Absolute field-centre LO weights, (2,2N).
        feedback_matrix: Loop feedback matrix B=sum(g_g R_g), (2,2).
        interpolation_feedback_matrix: Matrix used to evaluate the filtered
            covariance interpolators, (2,2). Matches the feedback matrix
            unless a symmetric approximation is used.
        interpolation_feedback_eigenvalues: Two real eigenvalues of the
            interpolation feedback matrix.
        interpolation_feedback_projectors: Corresponding, generally non-orthogonal
            projectors, (2,2,2).
        used_symmetric_feedback_approximation: Whether covariance interpolation
            used the symmetric approximation for complex eigenvalues.
        feedback_approximation_relative_error: Relative Frobenius change in B.
        eigenbasis_condition_number: Condition number of the chosen eigenvector basis.
        atmospheric_command_covariance: Cov(u_atm), (2,2).
        ho_command_covariance: Cov(u_HO), (2,2).
        atmospheric_ho_command_covariance: Cross-covariance of the atmospheric
            and HO commands, in that order, (2,2). Can be asymmetric.
        science_atmospheric_command_covariance: Cross-covariance of science
            tilt with the atmospheric command, (S,2,2).
        science_ho_command_covariance: Cross-covariance of science tilt with
            the HO command, (S,2,2).
        science_tilt_covariance: Atmospheric tilt covariance at science positions
            before low-order correction, (S,2,2).
        atmospheric_residual_covariance: Atmospheric residual covariance, (S,2,2).
        ho_correction: HO covariance correction including both cross terms,
            (S,2,2). It can be indefinite.
        noiseless_covariance: Residual covariance after low-order correction,
            before adding the noise-driven command covariance, (S,2,2).
        noise_command_covariance: Stationary covariance of the noise-driven
            command using actual B, (2,2).
        noise_covariance_scales: Scale factors for the noise-command covariance
            in the narrow and broad mixture components, (2,).
        distribution: Centred displacement distribution for the science field,
            with a narrow and broad Gaussian component in that order. Its
            ``covariance`` is the total image-displacement covariance.

    No per-component PSD clipping is applied. The model rejects a noiseless
    covariance with an eigenvalue below -1e-8 mas squared; tiny roundoff is
    retained explicitly rather than hidden by conditioning.
    """

    dtype: np.dtype
    photons_per_frame: u.Quantity
    ngs_metrics: PsfMetricPrediction
    coma_response_coefficients: u.Quantity
    sensing_gains: u.Quantity
    wfs_noise_variances: u.Quantity
    reconstructor_noise_covariance: u.Quantity
    reconstructor: np.ndarray
    feedback_matrix: np.ndarray
    interpolation_feedback_matrix: np.ndarray
    interpolation_feedback_eigenvalues: np.ndarray
    interpolation_feedback_projectors: np.ndarray
    used_symmetric_feedback_approximation: bool
    feedback_approximation_relative_error: float
    eigenbasis_condition_number: float
    atmospheric_command_covariance: u.Quantity
    ho_command_covariance: u.Quantity
    atmospheric_ho_command_covariance: u.Quantity
    science_atmospheric_command_covariance: u.Quantity
    science_ho_command_covariance: u.Quantity
    science_tilt_covariance: u.Quantity
    atmospheric_residual_covariance: u.Quantity
    ho_correction: u.Quantity
    noiseless_covariance: u.Quantity
    noise_command_covariance: u.Quantity
    noise_covariance_scales: np.ndarray
    distribution: LowOrderDistribution
    _total: u.Quantity = field(repr=False)

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
        "coma_response_coefficients": u.mas / u.nm,
        "sensing_gains": u.one,
        "wfs_noise_variances": u.mas**2,
        "reconstructor_noise_covariance": u.mas**2,
        **{
            name: u.mas**2
            for name in (
                "atmospheric_command_covariance",
                "ho_command_covariance",
                "atmospheric_ho_command_covariance",
                "science_atmospheric_command_covariance",
                "science_ho_command_covariance",
                "science_tilt_covariance",
                "atmospheric_residual_covariance",
                "ho_correction",
                "noiseless_covariance",
                "noise_command_covariance",
                "_total",
            )
        },
    }
    for name, value in values.items():
        if name in units:
            if not isinstance(value, u.Quantity):
                raise TypeError(f"{name} must be a quantity")
            value = _quantity(value, units[name], name, value.ndim, dtype=dtype)
        elif name == "distribution":
            if not isinstance(value, LowOrderDistribution):
                raise TypeError("distribution must be LowOrderDistribution")
            if value.dtype != dtype:
                raise ValueError("distribution precision must match the prediction")
        elif name == "ngs_metrics":
            if not isinstance(value, PsfMetricPrediction):
                raise TypeError("ngs_metrics must be PsfMetricPrediction")
            value = replace(value, dtype=dtype)
        elif name == "used_symmetric_feedback_approximation":
            if not isinstance(value, (bool, np.bool_)):
                raise TypeError("feedback fallback must be boolean")
            value = bool(value)
        elif name in (
            "feedback_approximation_relative_error",
            "eigenbasis_condition_number",
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
    n = len(result.ngs_metrics.ee)
    if result.science_tilt_covariance.ndim != 3:
        raise ValueError("science_tilt_covariance must be (science,2,2)")
    count = len(result.science_tilt_covariance)
    shapes = {
        **{
            key: (n,)
            for key in (
                "photons_per_frame",
                "coma_response_coefficients",
                "sensing_gains",
                "wfs_noise_variances",
            )
        },
        "reconstructor_noise_covariance": (2 * n, 2 * n),
        "reconstructor": (2, 2 * n),
        "feedback_matrix": (2, 2),
        "interpolation_feedback_matrix": (2, 2),
        "interpolation_feedback_eigenvalues": (2,),
        "interpolation_feedback_projectors": (2, 2, 2),
        "noise_covariance_scales": (2,),
        **{
            key: (2, 2)
            for key in (
                "atmospheric_command_covariance",
                "ho_command_covariance",
                "atmospheric_ho_command_covariance",
                "noise_command_covariance",
            )
        },
        **{
            key: (count, 2, 2)
            for key in (
                "science_atmospheric_command_covariance",
                "science_ho_command_covariance",
                "science_tilt_covariance",
                "atmospheric_residual_covariance",
                "ho_correction",
                "noiseless_covariance",
                "_total",
            )
        },
    }
    for name, shape in shapes.items():
        if getattr(result, name).shape != shape:
            raise ValueError(f"prediction {name.removeprefix('_')} must have shape {shape}")
    if result.distribution.covariances.shape != (count, 2, 2, 2):
        raise ValueError(
            "prediction distribution requires two components per science position"
        )
    if any(
        np.any(getattr(result, key).value <= 0)
        for key in (
            "photons_per_frame",
            "coma_response_coefficients",
            "sensing_gains",
            "wfs_noise_variances",
        )
    ):
        raise ValueError(
            "positive photon counts and WFS response/noise values required"
        )


def _validate_feedback(result: E2ELowOrderPrediction, tolerance: float) -> None:
    """Verify actual feedback and the scalar decomposition used by the interpolator."""
    if not np.allclose(
        result.feedback_matrix,
        _feedback_matrix(result.reconstructor, result.sensing_gains.value),
        rtol=tolerance,
        atol=tolerance,
    ):
        raise ValueError(
            "feedback_matrix must equal the gain-weighted reconstructor sum"
        )
    _validate_scalar_decomposition(
        result.feedback_matrix,
        result.interpolation_feedback_matrix,
        result.interpolation_feedback_eigenvalues,
        result.interpolation_feedback_projectors,
        result.used_symmetric_feedback_approximation,
    )


def _validate_covariance_components(
    result: E2ELowOrderPrediction, tolerance: float
) -> None:
    """Check the physical command and residual covariance identities."""
    for name in (
        "reconstructor_noise_covariance",
        "atmospheric_command_covariance",
        "ho_command_covariance",
        "science_tilt_covariance",
        "atmospheric_residual_covariance",
        "noiseless_covariance",
        "noise_command_covariance",
        "_total",
    ):
        value = getattr(result, name).value
        if (
            not np.allclose(value, value.swapaxes(-1, -2), rtol=tolerance, atol=1e-8)
            or np.linalg.eigvalsh(value).min() < -1e-8
        ):
            raise ValueError(
                f"prediction {name.removeprefix('_')} must be symmetric positive semidefinite"
            )

    terms = _TiltCommandCovariances(
        science_tilt=result.science_tilt_covariance.value,
        atmospheric_command=result.atmospheric_command_covariance.value,
        ho_command=result.ho_command_covariance.value,
        atmospheric_ho_command=result.atmospheric_ho_command_covariance.value,
        science_atmospheric_command=result.science_atmospheric_command_covariance.value,
        science_ho_command=result.science_ho_command_covariance.value,
    )
    atmospheric, ho = terms.residual_covariances()
    if not np.allclose(
        result.atmospheric_residual_covariance.value,
        atmospheric,
        rtol=tolerance,
        atol=1e-8,
    ):
        raise ValueError(
            "atmospheric residual covariance disagrees with stored command terms"
        )
    if not np.allclose(result.ho_correction.value, ho, rtol=tolerance, atol=1e-8):
        raise ValueError(
            "HO covariance contribution disagrees with stored command terms"
        )
    if not np.allclose(
        result.noiseless_covariance.value,
        result.atmospheric_residual_covariance.value + result.ho_correction.value,
        rtol=tolerance,
        atol=1e-8,
    ):
        raise ValueError("noiseless covariance must equal atmospheric plus HO")
    if not np.allclose(
        result._total.value,
        result.noiseless_covariance.value + result.noise_command_covariance.value,
        rtol=tolerance,
        atol=1e-8,
    ):
        raise ValueError("total covariance must equal noiseless plus noise")


def _validate_displacement_mixture(
    result: E2ELowOrderPrediction, tolerance: float
) -> None:
    """Check that the applied Gaussian mixture preserves the total covariance."""
    distribution = result.distribution
    if np.any(result.noise_covariance_scales <= 0) or not np.isclose(
        distribution.weights @ result.noise_covariance_scales, 1, rtol=tolerance
    ):
        raise ValueError("noise mixture must preserve covariance")
    expected = (
        result.noiseless_covariance.value[:, None]
        + result.noise_covariance_scales[None, :, None, None]
        * result.noise_command_covariance.value
    )
    covariances = distribution.covariances.value
    if not np.allclose(covariances, expected, rtol=tolerance, atol=1e-8):
        raise ValueError("mixture covariances differ from noiseless and scaled noise")
    if not np.allclose(
        covariances, covariances.swapaxes(-1, -2), rtol=tolerance, atol=1e-8
    ):
        raise ValueError("prediction mixture covariances must be symmetric")
