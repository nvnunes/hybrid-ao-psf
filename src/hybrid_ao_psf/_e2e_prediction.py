"""Low-order displacement and optional diagnostic groups for E2E prediction."""

from __future__ import annotations

from dataclasses import dataclass, fields, replace

import numpy as np
from ao_core.models.control import tip_tilt_feedback
from ao_core.models.residual import assemble_tilt_residual_covariances
from astropy import units as u
from numpy.typing import NDArray

from ._calibration_contracts import _quantity
from ._lo_model import (
    _check_psd_covariance,
    _ControlSolution,
    _LowOrderCovariances,
)
from ._sensing_response import SensingPrediction
from ._units import _PhysicalRecord
from .contracts import LowOrderDistribution, PsfMetricPrediction, _readonly_array


@dataclass(frozen=True, kw_only=True, eq=False)
class _NgsDiagnostics(_PhysicalRecord):
    """Evaluated per-star inputs to the calibrated prediction.

    Metrics and responses use NGS order. The noise prior uses interleaved
    x/y axes and is in mas squared. Sensing variances are per axis.
    """

    photons_per_frame: u.Quantity
    metrics: PsfMetricPrediction
    coma_response_coefficients: u.Quantity
    sensing: SensingPrediction
    reconstructor_noise_covariance: u.Quantity


@dataclass(frozen=True, kw_only=True, eq=False)
class E2ELowOrderDiagnostics:
    """Inputs and intermediate results retained by a debug prediction.

    Attributes:
        ngs: Photon counts, spot metrics, coma response, sensing gain and
            variance, and the noise prior used for reconstruction.
        control: Field-centre reconstructor, actual feedback, and the scalar
            responses used to evaluate the covariance interpolators.
        covariances: Command terms and assembled residual covariances in mas
            squared. Matrix axes are in physical x/y order.
        noise_covariance_scales: Noise covariance multipliers in narrow and
            broad Gaussian component order.

    Values are read-only snapshots of the calculation.
    """

    ngs: _NgsDiagnostics
    control: _ControlSolution
    covariances: _LowOrderCovariances
    noise_covariance_scales: NDArray

    def __post_init__(self) -> None:
        dtype = self.ngs.sensing.dtype
        ngs = replace(
            self.ngs,
            photons_per_frame=_quantity(
                self.ngs.photons_per_frame,
                u.photon,
                "photons_per_frame",
                1,
                dtype=dtype,
            ),
            coma_response_coefficients=_quantity(
                self.ngs.coma_response_coefficients,
                u.mas / u.nm,
                "coma_response_coefficients",
                1,
                dtype=dtype,
            ),
            reconstructor_noise_covariance=_quantity(
                self.ngs.reconstructor_noise_covariance,
                u.mas**2,
                "reconstructor_noise_covariance",
                2,
                dtype=dtype,
            ),
        )
        object.__setattr__(self, "ngs", ngs)
        object.__setattr__(self, "control", _copy_control(self.control, dtype))
        object.__setattr__(
            self, "covariances", _copy_covariances(self.covariances, dtype)
        )
        object.__setattr__(
            self,
            "noise_covariance_scales",
            _readonly_array(
                self.noise_covariance_scales,
                dtype=dtype,
                label="noise_covariance_scales",
                ndim=1,
            ),
        )


@dataclass(frozen=True, kw_only=True, eq=False)
class E2ELowOrderPrediction:
    """Calibrated low-order displacement at the requested science positions.

    Attributes:
        displacement_distribution: Centred Gaussian mixture in angular x/y
            coordinates, with narrow and broad components in that order.
        diagnostics: Evaluated inputs, control response and intermediate
            covariance terms when debug is requested.
    """

    displacement_distribution: LowOrderDistribution
    diagnostics: E2ELowOrderDiagnostics | None = None


def _copy_control(control: _ControlSolution, dtype: np.dtype) -> _ControlSolution:
    """Give the debug solution immutable copies of the numerical matrices."""
    feedback = control.scalar_feedback
    scalar_feedback = replace(
        feedback,
        effective_feedback=_readonly_array(
            feedback.effective_feedback, dtype=dtype, label="effective_feedback", ndim=2
        ),
        strengths=_readonly_array(
            feedback.strengths, dtype=dtype, label="strengths", ndim=1
        ),
        projectors=_readonly_array(
            feedback.projectors, dtype=dtype, label="projectors", ndim=3
        ),
    )
    return replace(
        control,
        reconstructor=_readonly_array(
            control.reconstructor, dtype=dtype, label="reconstructor", ndim=2
        ),
        feedback=_readonly_array(
            control.feedback, dtype=dtype, label="feedback", ndim=2
        ),
        scalar_feedback=scalar_feedback,
    )


def _copy_covariances(
    covariances: _LowOrderCovariances, dtype: np.dtype
) -> _LowOrderCovariances:
    """Preserve the solved covariance groups with immutable physical arrays."""
    terms = covariances.terms
    if terms.science_tilt.dtype != dtype:
        terms = replace(
            terms,
            **{
                item.name: getattr(terms, item.name).astype(dtype)
                for item in fields(terms)
            },
        )
    return _LowOrderCovariances(
        terms=terms,
        atmospheric_residual=_quantity(
            covariances.atmospheric_residual,
            u.mas**2,
            "atmospheric_residual",
            3,
            dtype=dtype,
        ),
        ho_correction=_quantity(
            covariances.ho_correction, u.mas**2, "ho_correction", 3, dtype=dtype
        ),
        noiseless=_quantity(
            covariances.noiseless, u.mas**2, "noiseless", 3, dtype=dtype
        ),
        noise_command=_quantity(
            covariances.noise_command, u.mas**2, "noise_command", 2, dtype=dtype
        ),
    )


def _check_noise_scales(weights: NDArray, scales: NDArray) -> None:
    """Require the narrow and broad noise components to preserve variance."""
    tolerance = max(1e-10, 32 * np.finfo(scales.dtype).eps)
    if np.any(scales <= 0) or not np.isclose(weights @ scales, 1, rtol=tolerance):
        raise ValueError("noise mixture must preserve covariance")


def _check_saved_prediction(prediction: E2ELowOrderPrediction) -> None:
    """Check relationships among independently stored debug fields on load."""
    diagnostics = prediction.diagnostics
    ngs, control, covariances = (
        diagnostics.ngs,
        diagnostics.control,
        diagnostics.covariances,
    )
    distribution = prediction.displacement_distribution
    tolerance = max(1e-10, 32 * np.finfo(distribution.dtype).eps)
    if not np.allclose(
        control.feedback,
        tip_tilt_feedback(control.reconstructor, ngs.sensing.gain.value),
        rtol=tolerance,
        atol=tolerance,
    ):
        raise ValueError(
            "feedback_matrix must equal the gain-weighted reconstructor sum"
        )
    scalar = control.scalar_feedback
    scalar.validate(control.feedback)

    terms = covariances.terms
    for name, covariance in (
        ("reconstructor_noise_covariance", ngs.reconstructor_noise_covariance),
        ("atmospheric_command_covariance", terms.atmospheric_command),
        ("ho_command_covariance", terms.ho_command),
        ("science_tilt_covariance", terms.science_tilt),
        ("atmospheric_residual_covariance", covariances.atmospheric_residual),
        ("noiseless_covariance", covariances.noiseless),
        ("noise_command_covariance", covariances.noise_command),
        ("total_covariance", covariances.total),
    ):
        _check_psd_covariance(covariance.value, name)
    residual = assemble_tilt_residual_covariances(
        terms, command_noise=covariances.noise_command
    )
    for actual, expected, error in (
        (
            covariances.atmospheric_residual.value,
            residual.atmospheric_residual.value,
            "atmospheric residual covariance disagrees with stored command terms",
        ),
        (
            covariances.ho_correction.value,
            residual.ho_contribution.value,
            "HO covariance contribution disagrees with stored command terms",
        ),
        (
            covariances.noiseless.value,
            covariances.atmospheric_residual.value + covariances.ho_correction.value,
            "noiseless covariance must equal atmospheric plus HO",
        ),
    ):
        if not np.allclose(actual, expected, rtol=tolerance, atol=1e-8):
            raise ValueError(error)
    if np.any(ngs.photons_per_frame.value <= 0) or np.any(
        ngs.coma_response_coefficients.value <= 0
    ):
        raise ValueError("positive photon counts and coma response required")

    scales = diagnostics.noise_covariance_scales
    _check_noise_scales(distribution.weights, scales)
    expected_mixture = (
        covariances.noiseless.value[:, None]
        + scales[None, :, None, None] * covariances.noise_command.value
    )
    if not np.allclose(
        distribution.covariances.value, expected_mixture, rtol=tolerance, atol=1e-8
    ):
        raise ValueError("mixture covariances differ from noiseless and scaled noise")
