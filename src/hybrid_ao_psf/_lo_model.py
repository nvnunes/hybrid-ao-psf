"""Control preparation and covariance diagnostics for low-order prediction."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from ao_core.models.control import FixedGainIntegrator, ScalarFeedback, TipTiltControl
from ao_core.models.reconstruction import lo_reconstructor
from ao_core.models.residual import TiltCommandCovariances, TiltResidualCovariances
from astropy import units as u
from numpy.typing import DTypeLike, NDArray

from ._units import _PhysicalRecord


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
class _ControlSolution:
    """Field-centre reconstructor, actual feedback and scalar filtering response."""

    reconstructor: NDArray
    feedback: NDArray
    scalar_feedback: ScalarFeedback

    @property
    def reconstructor_blocks(self) -> NDArray:
        """View R as (NGS, 2, 2) in interleaved physical x/y order."""
        return self.reconstructor.reshape(2, -1, 2).transpose(1, 0, 2)


@dataclass(frozen=True, kw_only=True, eq=False)
class _LowOrderCovariances(_PhysicalRecord):
    """Angular covariance terms and results retained in debug diagnostics."""

    terms: TiltCommandCovariances
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


def _solve_control(
    *,
    reconstruction_covariances: _ReconstructorCovariances | None,
    sensing_gains: u.Quantity,
    controller: FixedGainIntegrator,
    dtype: DTypeLike,
) -> TipTiltControl:
    """Solve the reconstructor and validate both feedback responses.

    Args:
        reconstruction_covariances: Atmospheric covariances and noise prior
            for the reconstructor. One NGS uses identity weights.
        sensing_gains: One dimensionless gain per NGS.
        controller: Integrator used to check the stability of actual and
            effective feedback.
        dtype: Precision of the calibrated reconstruction.

    Returns:
        The reconstructor, actual feedback and scalar feedback response.

    Raises:
        ValueError: If reconstruction inputs or feedback are invalid.
    """
    if sensing_gains.ndim != 1 or not sensing_gains.size:
        raise ValueError("sensing gains must be a nonempty NGS vector")

    ngs_count = len(sensing_gains)
    if ngs_count == 1:
        reconstructor = np.eye(2, dtype=dtype)
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
            dtype=dtype,
        )
    return TipTiltControl(
        reconstructor=reconstructor * u.one,
        sensing_gains=sensing_gains,
        controller=controller,
    )


def _check_prediction_covariances(
    terms: TiltCommandCovariances, residual: TiltResidualCovariances
) -> None:
    """Apply Hybrid's physical covariance checks at the prediction precision."""
    for name, covariance in (
        ("science_tilt_covariance", terms.science_tilt),
        ("atmospheric_command_covariance", terms.atmospheric_command),
        ("ho_command_covariance", terms.ho_command),
        ("atmospheric_residual_covariance", residual.atmospheric_residual),
    ):
        _check_psd_covariance(covariance.value, name)
    if np.linalg.eigvalsh(residual.noiseless.value).min() < -1e-8:
        raise ValueError("calibrated noiseless covariance is not positive semidefinite")
    _check_psd_covariance(residual.command_noise.value, "noise_command_covariance")
    _check_psd_covariance(residual.total.value, "total_covariance")
