"""Fixed-gain delayed-integrator response and stationary command covariance.

Numerical inputs use two angular tilt axes. Reconstructed sensing response and
controller gain are dimensionless; noise covariance uses the caller's squared
angular unit. No calibration bundle or E2E implementation is required.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.linalg import solve_discrete_lyapunov
from scipy.signal import lfilter, lfilter_zi

from ._validation import validate_psf_dtype


def _input_dtype(value, dtype):
    """Infer internal precision from prepared arrays; integers use float32."""
    if dtype is None:
        source = np.asarray(value).dtype
        dtype = source if source.kind == "f" else np.float32
    return validate_psf_dtype(dtype)


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
    if (
        isinstance(delay, (bool, np.bool_))
        or not isinstance(delay, (int, np.integer))
        or delay < 1
    ):
        raise ValueError("Delay must be a positive integer")
    response = np.asarray(response)
    if (
        response.dtype.kind not in "fiu"
        or response.shape != (2, 2)
        or not np.isfinite(response).all()
    ):
        raise ValueError(
            "Reconstructed sensing response must be a finite two-by-two matrix"
        )
    if not np.isfinite(kappa) or kappa <= 0:
        raise ValueError("Controller gain must be finite and positive")
    dtype = _input_dtype(response, dtype)
    response = response.astype(dtype, copy=False)
    kappa = dtype.type(kappa)
    transition = np.zeros((2 * delay, 2 * delay), dtype=dtype)
    transition[:2, :2] = np.eye(2, dtype=dtype)
    # For n=1 the current and delayed blocks coincide: I - kappa B.
    transition[:2, -2:] -= kappa * response
    transition[2:, :-2] = np.eye(2 * (delay - 1), dtype=dtype)
    radius = float(np.max(np.abs(np.linalg.eigvals(transition))))
    if radius >= 1:
        raise ValueError(
            f"Unstable LO loop: spectral radius rho(A)={radius:.12g} must be < 1 "
            f"(gain={kappa}, delay={delay})"
        )
    return transition


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
    dtype = _input_dtype(reconstructed_noise, dtype)
    transition = integrator_transition(response, kappa, delay, dtype=dtype)
    kappa = dtype.type(kappa)
    noise = np.asarray(reconstructed_noise, dtype=dtype)
    if (
        noise.shape != (2, 2)
        or not np.isfinite(noise).all()
        or not np.allclose(
            noise, noise.T, rtol=max(1e-12, 8 * np.finfo(dtype).eps), atol=1e-12
        )
    ):
        raise ValueError(
            "Reconstructed noise covariance must be finite, symmetric and two-by-two"
        )
    noise = (noise + noise.T) / 2
    if np.linalg.eigvalsh(noise).min() < -1e-12 * max(1.0, np.linalg.norm(noise)):
        raise ValueError("Reconstructed noise covariance must be positive semidefinite")
    innovation = np.zeros_like(transition)
    innovation[:2, :2] = kappa**2 * noise
    stationary = solve_discrete_lyapunov(transition, innovation)
    covariance = np.asarray(stationary[:2, :2], dtype=dtype)
    if not np.isfinite(covariance).all():
        raise ValueError("Nonfinite stationary LO noise covariance")
    return (covariance + covariance.T) / 2


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
    dtype = _input_dtype(css_mas2, dtype)
    css, cas, noise = [
        np.asarray(v, dtype=dtype)
        for v in (css_mas2, cas_center_mas2, centered_noise_mas2)
    ]
    n = css.shape[0] if css.ndim == 2 else 0
    if (
        n < 2
        or n % 2
        or css.shape != (n, n)
        or noise.shape != css.shape
        or cas.shape != (2, n)
    ):
        raise ValueError(
            "Expected Css and noise (2*N,2*N), and field-centre Cas (2,2*N)."
        )
    if not all(np.isfinite(v).all() for v in (css, cas, noise)):
        raise ValueError("Reconstructor covariances must be finite.")
    if n == 2:
        return np.eye(2, dtype=dtype)
    return np.linalg.solve((css + noise).T, cas.T).T


def state_matrix(
    gain: float, delay: int, strength: float, *, dtype=np.float64
) -> np.ndarray:
    """Chronological command history transition for the delayed integrator."""
    dtype = validate_psf_dtype(dtype)
    gain, strength = dtype.type(gain), dtype.type(strength)
    state = np.zeros((delay, delay), dtype=dtype)
    if delay > 1:
        state[:-1, 1:] = np.eye(delay - 1, dtype=dtype)
    state[-1, -1] += 1
    state[-1, 0] -= gain * strength
    return state


def settling_cut(
    gain: float,
    delay: int,
    strengths,
    *,
    tolerance: float = 1e-8,
    maximum: int = 500,
    dtype=np.float64,
) -> tuple[int, float]:
    """Find the common cut bounding initial-state propagation for every node."""
    states = [state_matrix(gain, delay, b, dtype=dtype) for b in strengths]
    if any(max(abs(np.linalg.eigvals(a))) >= 1 for a in states):
        raise ValueError("Unstable feedback grid")
    for cut in range(delay, maximum + 1):
        envelope = max(
            np.linalg.norm(np.linalg.matrix_power(a, cut), 2) for a in states
        )
        if envelope <= tolerance:
            return cut, float(envelope)
    raise ValueError(f"No settled replay window within {maximum} saved frames")


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
    dtype = _input_dtype(history, dtype)
    history = np.asarray(history, dtype=dtype)
    gain, strength = dtype.type(gain), dtype.type(strength)
    numerator = np.zeros(delay + 1, dtype=dtype)
    numerator[-1] = gain
    denominator = np.zeros(delay + 1, dtype=dtype)
    denominator[0], denominator[1] = 1.0, -1.0
    denominator[-1] += gain * strength
    if not equilibrium:
        return lfilter(numerator, denominator, history, axis=-1)
    zi = history[..., :1] * lfilter_zi(numerator, denominator)
    return lfilter(numerator, denominator, history, axis=-1, zi=zi)[0]


def filter_input(
    x: np.ndarray, B: np.ndarray, gain: float, delay: int, *, dtype=None
) -> np.ndarray:
    """Zero-history delayed integrator response to shape (2,realizations,frames).

    A complex Schur basis is a numerically stable exact 2D decomposition.
    Coupled off-diagonal response is retained; no symmetric-B approximation or
    potentially ill-conditioned eigenvector inversion enters this replay.
    """
    from scipy.linalg import schur

    dtype = _input_dtype(x, dtype)
    complex_dtype = np.complex64 if dtype.itemsize == 4 else np.complex128
    x = np.asarray(x, dtype=dtype)
    B = np.asarray(B, dtype=dtype)
    gain = dtype.type(gain)
    integrator_transition(B, gain, delay, dtype=dtype)
    triangular, basis = schur(B, output="complex")
    triangular, basis = triangular.astype(complex_dtype), basis.astype(complex_dtype)
    y = np.einsum("ab,brt->art", basis.conj().T, x)
    numerator = np.r_[np.zeros(delay, dtype=dtype), gain]
    denominator2 = np.r_[1.0, -1.0, np.zeros(max(delay - 1, 0), dtype=dtype)].astype(
        complex_dtype
    )
    denominator2[delay] += gain * triangular[1, 1]
    second = lfilter(numerator, denominator2, y[1], axis=-1)
    first_input = y[0] - triangular[0, 1] * second
    denominator1 = np.r_[1.0, -1.0, np.zeros(max(delay - 1, 0), dtype=dtype)].astype(
        complex_dtype
    )
    denominator1[delay] += gain * triangular[0, 0]
    first = lfilter(numerator, denominator1, first_input, axis=-1)
    result = np.einsum("ab,brt->art", basis, np.stack((first, second)))
    bound = 32 * np.finfo(dtype).eps * max(1.0, float(np.max(np.abs(result.real))))
    if np.max(np.abs(result.imag)) > bound:
        raise ValueError("Noise replay has a nonreal numerical remainder.")
    return result.real
