"""Control solution and residual covariance for low-order prediction."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from ao_core.models.control import FixedGainIntegrator
from astropy import units as u
from numpy.typing import DTypeLike, NDArray

from ._calibration_numerics import lo_reconstructor
from ._units import _PhysicalRecord, quantity_from_value


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
class _TiltCommandCovariances(_PhysicalRecord):
    """Science tilt and filtered command covariances.

    Each matrix is in mas² and uses x/y on both axes. Terms involving science
    tilt have shape (science, 2, 2); command-only terms have shape (2, 2).
    A cross-covariance lists its arguments in row, column order and can be
    asymmetric.
    """

    science_tilt: u.Quantity
    atmospheric_command: u.Quantity
    ho_command: u.Quantity
    atmospheric_ho_command: u.Quantity
    science_atmospheric_command: u.Quantity
    science_ho_command: u.Quantity

    def assemble_residual_covariances(self) -> tuple[NDArray, NDArray]:
        """Assemble atmospheric residual covariance and the HO contribution."""
        science_atmospheric_command = self.science_atmospheric_command.value
        atmospheric_residual_covariance = (
            self.science_tilt.value
            + self.atmospheric_command.value
            - science_atmospheric_command
            - science_atmospheric_command.swapaxes(-1, -2)
        )
        # Cov(t_atm - u_atm, u_HO) retains the order of its two arguments.
        residual_ho_cross_covariance = (
            self.science_ho_command.value - self.atmospheric_ho_command.value
        )
        ho_covariance_contribution = (
            self.ho_command.value
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
class _LowOrderCovariances(_PhysicalRecord):
    """Terms used to assemble the angular residual covariance, in mas²."""

    terms: _TiltCommandCovariances
    atmospheric_residual: u.Quantity
    ho_correction: u.Quantity
    noiseless: u.Quantity
    noise_command: u.Quantity

    @property
    def total(self) -> u.Quantity:
        """Total angular covariance from the noiseless and noise terms."""
        return self.noiseless + self.noise_command


def _check_psd_covariance(matrix: NDArray, name: str) -> None:
    """Require a finite, symmetric covariance with the model's PSD tolerance."""
    tolerance = max(1e-10, 32 * np.finfo(matrix.dtype).eps)
    if (
        not np.isfinite(matrix).all()
        or not np.allclose(matrix, matrix.swapaxes(-1, -2), rtol=tolerance, atol=1e-8)
        or np.linalg.eigvalsh(matrix).min() < -1e-8
    ):
        raise ValueError(f"{name} must be symmetric positive semidefinite")


def _reconstructor_blocks(reconstructor: NDArray) -> NDArray:
    return reconstructor.reshape(2, -1, 2).transpose(1, 0, 2)


def _compute_feedback_matrix(reconstructor: NDArray, sensing_gains: NDArray) -> NDArray:
    """Compute B from the gain-weighted reconstructor blocks."""
    return np.einsum("g,gab->ab", sensing_gains, _reconstructor_blocks(reconstructor))


def _compute_noise_command_covariance(
    controller: FixedGainIntegrator,
    feedback: NDArray,
    reconstructor_blocks: NDArray,
    wfs_noise_variances: NDArray,
) -> u.Quantity:
    """Propagate reconstructed WFS noise through the actual matrix feedback.

    Reconstructor blocks are in (NGS, x/y, x/y) order. The independent
    per-axis WFS variances and the returned command covariance are in mas².
    """
    if wfs_noise_variances.shape != (len(reconstructor_blocks),):
        raise ValueError("WFS noise variances must align with the reconstructor")
    reconstructed_noise = np.einsum(
        "g,gab,gcb->ac",
        wfs_noise_variances,
        reconstructor_blocks,
        reconstructor_blocks,
    )
    return controller.noise_covariance(feedback, reconstructed_noise * u.mas**2)


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


def _decompose_feedback(feedback: NDArray) -> _ScalarFeedback:
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

    Concrete models evaluate command covariances using the solved control
    response and build the displacement distribution from the angular residual.
    """

    dtype: DTypeLike

    def _solve_control(
        self,
        *,
        reconstruction_covariances: _ReconstructorCovariances | None,
        sensing_gains: NDArray,
        controller: FixedGainIntegrator,
    ) -> _ControlSolution:
        """Solve the reconstructor and validate both feedback responses.

        Args:
            reconstruction_covariances: Atmospheric covariances and noise prior
                for the reconstructor. One NGS uses identity weights.
            sensing_gains: One dimensionless gain per NGS.
            controller: Integrator used to check the stability of actual and
                effective feedback.

        Returns:
            The reconstructor, actual feedback and scalar feedback response.

        Raises:
            ValueError: If reconstruction inputs or feedback are invalid.
        """
        if sensing_gains.ndim != 1 or not sensing_gains.size:
            raise ValueError("sensing gains must be a nonempty NGS vector")

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
        feedback = _compute_feedback_matrix(reconstructor, sensing_gains)
        controller.transition(feedback)
        scalar_feedback = _decompose_feedback(feedback)
        controller.transition(scalar_feedback.effective_feedback)
        return _ControlSolution(
            reconstructor=reconstructor,
            feedback=feedback,
            scalar_feedback=scalar_feedback,
        )

    @staticmethod
    def _assemble_low_order_covariances(
        *,
        command_covariances: _TiltCommandCovariances,
        control: _ControlSolution,
        wfs_noise_variances: NDArray,
        controller: FixedGainIntegrator,
    ) -> _LowOrderCovariances:
        """Assemble the angular residual and propagate WFS measurement noise.

        Args:
            command_covariances: Science tilt and filtered command terms in mas².
            control: Solved reconstructor and feedback for the same asterism.
            wfs_noise_variances: One per-axis variance in mas² per NGS.
            controller: Integrator used to propagate noise with actual feedback.

        Returns:
            Atmospheric, HO, noiseless and noise-command covariance terms in mas².

        Raises:
            ValueError: If the variances do not align with the reconstructor or
                the covariance terms are nonphysical.
        """
        atmospheric_residual_covariance, ho_covariance_contribution = (
            command_covariances.assemble_residual_covariances()
        )
        for name, matrix in (
            ("science_tilt_covariance", command_covariances.science_tilt.value),
            (
                "atmospheric_command_covariance",
                command_covariances.atmospheric_command.value,
            ),
            ("ho_command_covariance", command_covariances.ho_command.value),
            ("atmospheric_residual_covariance", atmospheric_residual_covariance),
        ):
            _check_psd_covariance(matrix, name)
        noiseless_covariance = (
            atmospheric_residual_covariance + ho_covariance_contribution
        )
        if np.linalg.eigvalsh(noiseless_covariance).min() < -1e-8:
            raise ValueError(
                "calibrated noiseless covariance is not positive semidefinite"
            )

        noise_command_covariance = _compute_noise_command_covariance(
            controller,
            control.feedback,
            control.reconstructor_blocks,
            wfs_noise_variances,
        )
        _check_psd_covariance(
            noise_command_covariance.value, "noise_command_covariance"
        )
        _check_psd_covariance(
            noiseless_covariance + noise_command_covariance.value, "total_covariance"
        )
        return _LowOrderCovariances(
            terms=command_covariances,
            atmospheric_residual=quantity_from_value(
                atmospheric_residual_covariance, u.mas**2
            ),
            ho_correction=quantity_from_value(ho_covariance_contribution, u.mas**2),
            noiseless=quantity_from_value(noiseless_covariance, u.mas**2),
            noise_command=noise_command_covariance,
        )
