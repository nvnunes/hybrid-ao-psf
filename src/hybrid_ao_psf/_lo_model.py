"""Control solution and residual covariance for low-order prediction."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from ao_models import FixedGainIntegrator
from astropy import units as u
from numpy.typing import DTypeLike, NDArray

from ._calibration_numerics import lo_reconstructor


@dataclass(frozen=True, kw_only=True, eq=False)
class _ReconstructorCovariances:
    """Multi-NGS inputs for R = C_as(C_ss + C_eta)^-1.

    The NGS covariance and pre-filter noise prior are (2N, 2N). The
    field-centre/NGS covariance is (2, 2N). NGS axes interleave x and y.
    """

    ngs_covariance: u.Quantity
    centre_ngs_covariance: u.Quantity
    noise_prior: u.Quantity


@dataclass(frozen=True, kw_only=True, eq=False)
class _ScalarFeedback:
    """Effective feedback and its two scalar responses for covariance filtering."""

    effective_feedback: NDArray
    strengths: NDArray
    projectors: NDArray
    symmetric_fallback: bool
    approximation_relative_error: float
    eigenbasis_condition: float


@dataclass(frozen=True, kw_only=True, eq=False)
class _ControlSolution:
    """Field-centre reconstructor, actual feedback and scalar filtering response."""

    reconstructor: NDArray
    feedback: NDArray
    scalar_feedback: _ScalarFeedback

    @property
    def reconstructor_blocks(self) -> NDArray:
        """View R as (NGS, 2, 2) in interleaved physical x/y order."""
        return _reconstructor_blocks(self.reconstructor)


@dataclass(frozen=True, kw_only=True, eq=False)
class _TiltCommandCovariances:
    """Science tilt and filtered commands in canonical numerical units.

    Each matrix is in mas² and uses x/y on both axes. Terms involving science
    tilt have shape (science, 2, 2); command-only terms have shape (2, 2).
    A cross-covariance lists its arguments in row, column order and can be
    asymmetric.
    """

    science_tilt: NDArray
    atmospheric_command: NDArray
    ho_command: NDArray
    atmospheric_ho_command: NDArray
    science_atmospheric_command: NDArray
    science_ho_command: NDArray

    def residual_covariances(self) -> tuple[NDArray, NDArray]:
        """Assemble atmospheric residual covariance and the HO contribution."""
        science_atmospheric_command = self.science_atmospheric_command
        atmospheric_residual_covariance = (
            self.science_tilt
            + self.atmospheric_command
            - science_atmospheric_command
            - science_atmospheric_command.swapaxes(-1, -2)
        )
        # Cov(t_atm - u_atm, u_HO) retains the order of its two arguments.
        residual_ho_cross_covariance = (
            self.science_ho_command - self.atmospheric_ho_command
        )
        ho_covariance_contribution = (
            self.ho_command
            - residual_ho_cross_covariance
            - residual_ho_cross_covariance.swapaxes(-1, -2)
        )
        atmospheric_residual_covariance = (
            atmospheric_residual_covariance
            + atmospheric_residual_covariance.swapaxes(-1, -2)
        ) / 2
        ho_covariance_contribution = (
            ho_covariance_contribution + ho_covariance_contribution.swapaxes(-1, -2)
        ) / 2
        return atmospheric_residual_covariance, ho_covariance_contribution


@dataclass(frozen=True, kw_only=True, eq=False)
class _LowOrderCovariances:
    """Closed-loop terms and residual image-motion covariance in physical x/y."""

    control: _ControlSolution
    terms: _TiltCommandCovariances
    atmospheric_residual_covariance: u.Quantity
    ho_covariance_contribution: u.Quantity
    noiseless_covariance: u.Quantity
    noise_command_covariance: u.Quantity
    total_covariance: u.Quantity


def _reconstructor_blocks(reconstructor: NDArray) -> NDArray:
    return reconstructor.reshape(2, -1, 2).transpose(1, 0, 2)


def _feedback_matrix(reconstructor: NDArray, sensing_gains: NDArray) -> NDArray:
    """Compute B from the gain-weighted reconstructor blocks."""
    return np.einsum("g,gab->ab", sensing_gains, _reconstructor_blocks(reconstructor))


def _validate_scalar_decomposition(
    feedback: NDArray,
    effective_feedback: NDArray,
    strengths: NDArray,
    projectors: NDArray,
    symmetric_fallback: bool,
) -> None:
    """Check the scalar representation shared by runtime prediction and loading.

    The effective matrix must equal the actual feedback, or its symmetric part
    when the fallback is declared. Projectors must be idempotent, resolve identity
    and reproduce that matrix with the stored strengths. The idempotence check
    allows two-term matrix-product roundoff for non-orthogonal projectors.
    Values are checked, never conditioned.
    """
    tolerance = max(1e-10, 32 * np.finfo(feedback.dtype).eps)
    expected = (feedback + feedback.T) / 2 if symmetric_fallback else feedback
    if not np.allclose(effective_feedback, expected, rtol=tolerance, atol=tolerance):
        raise ValueError("effective feedback must match the declared feedback response")
    if not np.allclose(
        projectors.sum(0), np.eye(2), rtol=tolerance, atol=tolerance
    ) or not np.allclose(
        np.einsum("p,pab->ab", strengths, projectors),
        effective_feedback,
        rtol=tolerance,
        atol=tolerance,
    ):
        raise ValueError("feedback scalar decomposition failed reconstruction")
    eps = np.finfo(feedback.dtype).eps
    # Each 2x2 product entry is a two-term dot product, with error bounded
    # componentwise by gamma_2 * (abs(P) @ abs(P)).
    gamma_2 = 2 * eps / (1 - 2 * eps)
    with np.errstate(over="ignore", invalid="ignore"):
        products = projectors @ projectors
        product_roundoff = gamma_2 * (np.abs(projectors) @ np.abs(projectors))
        allowed = tolerance + tolerance * np.abs(projectors) + product_roundoff
    if (
        not np.isfinite(products).all()
        or not np.isfinite(allowed).all()
        or not np.all(np.abs(products - projectors) <= allowed)
    ):
        raise ValueError("feedback scalar decomposition requires idempotent projectors")


def _scalar_decomposition(feedback: NDArray) -> _ScalarFeedback:
    """Resolve real projectors, preserving the accepted complex-B approximation.

    A defective or excessively ill-conditioned eigenbasis cannot represent the
    calibrated scalar covariance interpolators. B itself remains unchanged for noise propagation.
    """
    dtype = feedback.dtype
    eps = np.finfo(dtype).eps
    values, basis = np.linalg.eig(feedback)
    fallback = bool(np.max(np.abs(values.imag)) > 1e-10)
    effective = (feedback + feedback.T) / 2 if fallback else feedback
    if fallback:
        values, basis = np.linalg.eigh(effective)
    else:
        values, basis = values.real, basis.real
    condition = float(np.linalg.cond(basis))
    if condition > min(1e10, 1 / (100 * eps)):
        if np.allclose(
            effective,
            np.eye(2, dtype=dtype) * np.trace(effective) / 2,
            rtol=max(1e-10, 4 * eps),
            atol=max(1e-12, eps),
        ):
            basis = np.eye(2, dtype=dtype)
            values = np.repeat(np.trace(effective) / 2, 2)
            condition = 1.0
        else:
            raise ValueError("feedback has an ill-conditioned or defective eigenbasis")
    inverse = np.linalg.inv(basis)
    projections = np.stack([np.outer(basis[:, p], inverse[p]) for p in range(2)])
    _validate_scalar_decomposition(feedback, effective, values, projections, fallback)
    return _ScalarFeedback(
        strengths=values,
        projectors=projections,
        effective_feedback=effective,
        symmetric_fallback=fallback,
        approximation_relative_error=float(
            np.linalg.norm(effective - feedback) / max(np.linalg.norm(feedback), 1e-30)
        ),
        eigenbasis_condition=condition,
    )


class _LowOrderModel:
    """Share the control solve and residual covariance assembly across models.

    The concrete model supplies the reconstructor inputs, WFS response,
    controller, and a command-covariance evaluator. It owns the displacement
    distribution formed from the resulting angular covariances.
    """

    dtype: DTypeLike

    def _predict_low_order(
        self,
        *,
        reconstruction_covariances: _ReconstructorCovariances | None,
        sensing_gains: NDArray,
        wfs_noise_variances: NDArray,
        controller: FixedGainIntegrator,
        evaluate_covariances: Callable[[_ControlSolution], _TiltCommandCovariances],
    ) -> _LowOrderCovariances:
        """Solve the loop, then assemble the noiseless residual and noise covariance.

        Args:
            reconstruction_covariances: Atmospheric covariance and noise prior
                for the reconstructor. One NGS uses identity weights and
                requires no reconstruction covariances.
            sensing_gains: One dimensionless gain per NGS.
            wfs_noise_variances: One per-axis variance in mas² per NGS.
            controller: Fixed integrator used for feedback and noise propagation.
            evaluate_covariances: Model-specific evaluator called once with
                the solved reconstructor, feedback matrix, and scalar
                decomposition. It borrows that solution without modifying it
                and returns atmospheric and HO terms in mas².

        Returns:
            The solved control response, model covariance terms, and assembled
            residual covariances. Noise propagation uses the actual matrix feedback.

        Raises:
            ValueError: If the inputs, feedback, or noiseless covariance are
                invalid, or the evaluator rejects an unsupported query.
        """
        if (
            sensing_gains.ndim != 1
            or not sensing_gains.size
            or wfs_noise_variances.shape != sensing_gains.shape
        ):
            raise ValueError(
                "sensing gains and WFS variances must be aligned NGS vectors"
            )

        # Solve the reconstructor and the feedback response.
        ngs_count = len(sensing_gains)
        if ngs_count == 1:
            reconstructor = np.eye(2, dtype=self.dtype)
        else:
            if reconstruction_covariances is None:
                raise ValueError(
                    "multiple NGS require atmospheric reconstruction covariances"
                )
            covariances = reconstruction_covariances
            if covariances.ngs_covariance.shape != (2 * ngs_count, 2 * ngs_count):
                raise ValueError("reconstructor covariance must have two axes per NGS")
            reconstructor = lo_reconstructor(
                covariances.ngs_covariance.to_value(u.mas**2),
                covariances.centre_ngs_covariance.to_value(u.mas**2),
                covariances.noise_prior.to_value(u.mas**2),
                dtype=self.dtype,
            )
        feedback = _feedback_matrix(reconstructor, sensing_gains)
        controller.transition(feedback)
        scalar_feedback = _scalar_decomposition(feedback)
        controller.transition(scalar_feedback.effective_feedback)
        control = _ControlSolution(
            reconstructor=reconstructor,
            feedback=feedback,
            scalar_feedback=scalar_feedback,
        )

        # Combine the model's atmospheric and HO terms into the noiseless residual.
        terms = evaluate_covariances(control)
        atmospheric_residual_covariance, ho_covariance_contribution = (
            terms.residual_covariances()
        )
        noiseless_covariance = (
            atmospheric_residual_covariance + ho_covariance_contribution
        )
        if np.linalg.eigvalsh(noiseless_covariance).min() < -1e-8:
            raise ValueError(
                "calibrated noiseless covariance is not positive semidefinite"
            )

        # Propagate reconstructed WFS noise with the actual matrix feedback.
        reconstructor_blocks = control.reconstructor_blocks
        reconstructed_noise_covariance = np.einsum(
            "g,gab,gcb->ac",
            wfs_noise_variances,
            reconstructor_blocks,
            reconstructor_blocks,
        )
        noise_command_covariance = controller.noise_covariance(
            feedback, reconstructed_noise_covariance * u.mas**2
        )
        noiseless_covariance = noiseless_covariance * u.mas**2
        return _LowOrderCovariances(
            control=control,
            terms=terms,
            atmospheric_residual_covariance=atmospheric_residual_covariance * u.mas**2,
            ho_covariance_contribution=ho_covariance_contribution * u.mas**2,
            noiseless_covariance=noiseless_covariance,
            noise_command_covariance=noise_command_covariance,
            total_covariance=noiseless_covariance + noise_command_covariance,
        )
