"""Compatibility entry points for the shared calibration kernels."""

from __future__ import annotations

import numpy as np
from ao_models import filter_input as _shared_filter_input
from ao_models import integrator_noise_covariance as _shared_integrator_noise_covariance
from ao_models import integrator_transition as _shared_integrator_transition
from ao_models import lo_reconstructor as _shared_lo_reconstructor
from ao_models import scalar_filter as _shared_scalar_filter
from ao_models import settling_cut, state_matrix
from numpy.typing import ArrayLike, NDArray


def integrator_transition(
    response: ArrayLike, kappa: float, delay: int, *, dtype=None
) -> NDArray:
    """Build the stable 2n-by-2n command-history transition matrix.

    ``response`` is the finite two-by-two reconstructed sensing response B=RG;
    ``kappa`` is the positive controller gain. ``delay`` is the positive integer
    measurement-to-command delay in frames. The command history runs from
    newest to oldest. Its first block obeys u[k]=u[k-1]-kappa*B*u[k-delay]
    in the absence of inputs. Invalid inputs or spectral radius >=1 raise
    ValueError. Input arrays are not mutated.
    """
    return _shared_integrator_transition(response, kappa, delay, dtype=dtype)


def integrator_noise_covariance(
    response: ArrayLike,
    reconstructed_noise: ArrayLike,
    kappa: float,
    delay: int,
    *,
    dtype=None,
) -> NDArray:
    """Return the stationary two-by-two noise-driven command covariance.

    ``response``, ``kappa`` and ``delay`` follow ``integrator_transition``.
    ``reconstructed_noise`` is a finite symmetric PSD two-by-two covariance
    of reconstructed measurement noise, independent between frames. The output
    uses the same squared angular unit. Noise enters the command after ``delay``
    frames; stationarity permits indexing that white innovation at the update.

    Solve S=A*S*A.T+Q with Q's first block kappa**2*reconstructed_noise and
    all other blocks zero, then return S's first two-by-two block. Invalid
    inputs or an unstable loop raise ValueError. Inputs are not mutated.
    """
    return _shared_integrator_noise_covariance(
        response, reconstructed_noise, kappa, delay, dtype=dtype
    )


def lo_reconstructor(
    css_mas2: np.ndarray,
    cas_center_mas2: np.ndarray,
    centered_noise_mas2: np.ndarray,
    *,
    dtype=None,
) -> np.ndarray:
    """Return absolute R = Cas(0,0) (Css + C_eta)^-1, or I for one NGS.

    Css and C_eta are 2*N square; Cas(0,0) is 2 by 2*N. All covariances
    are supplied in mas². This solver preserves the noise-dependent amplitude
    of R without conditioning matrices or normalizing reconstructor blocks.
    """
    return _shared_lo_reconstructor(
        css_mas2, cas_center_mas2, centered_noise_mas2, dtype=dtype
    )


def scalar_filter(
    history: np.ndarray,
    strength: float,
    gain: float,
    delay: int,
    *,
    equilibrium: bool = True,
    dtype=None,
) -> np.ndarray:
    """Filter (..., frames) input with a delayed integrator; preserve frame order.

    The response solves u[k]=u[k-1]-gain*b*u[k-delay]+gain*x[k-delay].
    Equilibrium initialization supplies constant first-input prehistory.
    Zero initialization is exposed only for independent settling checks.
    """
    return _shared_scalar_filter(
        history, strength, gain, delay, equilibrium=equilibrium, dtype=dtype
    )


def filter_input(
    x: np.ndarray, B: np.ndarray, gain: float, delay: int, *, dtype=None
) -> np.ndarray:
    """Zero-history delayed integrator response to shape (2,realizations,frames).

    A complex Schur basis is a numerically stable exact 2D decomposition.
    Coupled off-diagonal response is retained; no symmetric-B approximation or
    potentially ill-conditioned eigenvector inversion enters this simulation.
    """
    return _shared_filter_input(x, B, gain, delay, dtype=dtype)


__all__ = [
    "filter_input",
    "integrator_noise_covariance",
    "integrator_transition",
    "lo_reconstructor",
    "scalar_filter",
    "settling_cut",
    "state_matrix",
]
