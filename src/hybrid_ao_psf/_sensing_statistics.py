"""Native-free WFS moments, pooled replay statistics and noise-shape fitting."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize
from scipy.special import ndtr

from ._noise_blur import NoiseBlurModel
from ._validation import validate_psf_dtype

RADIAL_EDGES = np.r_[0.0, np.geomspace(1e-4, 200.0, 512), np.inf]
PROJECTION_EDGES = np.linspace(-8, 8, 257)
DIRECTIONS = np.column_stack(
    (np.cos(np.arange(4) * np.pi / 4), np.sin(np.arange(4) * np.pi / 4))
)


@dataclass(frozen=True)
class ReplayStatistics:
    """Shape diagnostics after pooled centring and whitening.

    Radial probabilities include the final infinite-tail bin and sum to one.
    Projection CDFs use four orientations at PROJECTION_EDGES. The empirical
    covariance retains differences between realization mean positions;
    production width still comes from the supplied Lyapunov covariance.
    """

    radial_probability: NDArray
    projection_cdf: NDArray
    radial_fourth: float
    covariance_mas2: NDArray
    sample_count: int


@dataclass(frozen=True)
class ReplayCase:
    """One retained asterism summarized using every captured replay sample."""

    key: str
    ngs_count: int
    population: str
    statistics: ReplayStatistics


def pooled_replay_moments(
    displacements_mas: NDArray,
    *,
    dtype=None,
) -> tuple[NDArray, NDArray, NDArray]:
    """Centre one case across all retained frames and realizations.

    Accept finite histories of shape (2,realizations,frames), with at least one
    realization and three retained frames. Return centred histories of that
    shape, the two-component pooled mean, and its 2-by-2 population covariance.
    Differences between realization means remain in the centred distribution.
    """
    dtype = validate_psf_dtype(
        np.asarray(displacements_mas).dtype if dtype is None else dtype
    )
    samples = np.asarray(displacements_mas, dtype=dtype)
    if (
        samples.ndim != 3
        or samples.shape[0] != 2
        or samples.shape[1] < 1
        or samples.shape[2] < 3
        or not np.isfinite(samples).all()
    ):
        raise ValueError("Replay must be finite with shape (2,realizations,frames).")
    mean = samples.mean(axis=(1, 2), keepdims=True)
    centred = samples - mean
    flat = centred.reshape(2, -1)
    covariance = flat @ flat.T / flat.shape[1]
    return centred, mean[:, 0, 0], covariance


def replay_statistics(displacements_mas: NDArray, *, dtype=None) -> ReplayStatistics:
    """Summarize shape after centring a (2,realizations,frames) case as one pool.

    The same pooled covariance both describes the displacement distribution and
    whitens it for shape fitting. Absolute prediction width comes from Lyapunov.
    """
    samples, _, covariance = pooled_replay_moments(displacements_mas, dtype=dtype)
    dtype = samples.dtype
    samples = samples.reshape(2, -1)
    white = np.linalg.solve(np.linalg.cholesky(covariance), samples).T
    radius = np.sum(white**2, axis=1)
    probability = np.histogram(radius, RADIAL_EDGES.astype(dtype))[0].astype(
        dtype
    ) / len(radius)
    cdfs = []
    for direction in DIRECTIONS.astype(dtype):
        values = white @ direction
        counts = np.histogram(
            values, np.r_[-np.inf, PROJECTION_EDGES.astype(dtype), np.inf]
        )[0]
        cdfs.append(np.cumsum(counts)[:-1].astype(dtype) / len(values))
    return ReplayStatistics(
        probability, np.array(cdfs), float(np.mean(radius**2)), covariance, len(radius)
    )


def _bin_probability(model: NoiseBlurModel, count: int) -> NDArray:
    narrow, broad = model.scales(count)
    weight = model.mixture_weight
    low = RADIAL_EDGES[:-1].astype(model.dtype)
    width = np.diff(RADIAL_EDGES.astype(model.dtype))

    def component(scale):
        return np.exp(-low / (2 * scale)) * -np.expm1(-width / (2 * scale))

    return weight * component(narrow) + (1 - weight) * component(broad)


def _bin_probability_gradient(model: NoiseBlurModel, count: int) -> NDArray:
    """Differentiate bin mass by mixture weight and the family's narrow scale.

    The broad scale is constrained by the unit total covariance. Handle the
    infinite final bin explicitly so its derivative never forms infinity times
    zero. Columns follow weight and narrow scale, respectively.
    """
    low = RADIAL_EDGES[:-1].astype(model.dtype)
    width = np.diff(RADIAL_EDGES.astype(model.dtype))
    weight = model.mixture_weight
    narrow, broad = model.scales(count)

    def component(scale):
        survival = np.exp(-low / (2 * scale))
        fraction = -np.expm1(-width / (2 * scale))
        upper_term = np.zeros_like(width)
        finite = np.isfinite(width)
        upper_term[finite] = width[finite] * np.exp(-width[finite] / (2 * scale))
        derivative = survival * (low * fraction - upper_term) / (2 * scale**2)
        return survival * fraction, derivative

    first, first_prime = component(narrow)
    second, second_prime = component(broad)
    return np.stack(
        (
            first - second + second_prime * (1 - narrow) / (1 - weight),
            weight * (first_prime - second_prime),
        ),
        axis=1,
    )


def fit_noise_blur(
    cases: list[ReplayCase], *, dtype=np.float32
) -> tuple[NoiseBlurModel, dict]:
    """Fit the documented three-coefficient covariance-preserving mixture.

    Equal total objective weight is assigned to one-NGS and multiple-NGS
    families; cases within each family share its weight equally. The objective
    is binned likelihood of squared Mahalanobis radius for centred 2D Gaussians.
    An analytic gradient includes the broad scale's covariance constraint.
    No PSF values, probe covariances or sensing-gain parameters enter this fit.
    """
    dtype = validate_psf_dtype(dtype)
    groups = {
        name: [c for c in cases if (c.ngs_count == 1) == single]
        for name, single in [("single", True), ("multi", False)]
    }
    if any(not members for members in groups.values()):
        raise ValueError("Mixture fitting requires single- and multiple-NGS cases.")
    pooled = {
        name: np.mean([c.statistics.radial_probability for c in members], axis=0)
        for name, members in groups.items()
    }

    def objective(values):
        model = NoiseBlurModel(*values, dtype=dtype)
        return -0.5 * sum(
            pooled[name]
            @ np.log(
                np.maximum(
                    _bin_probability(model, count),
                    1e-300 if dtype.itemsize == 8 else np.finfo(dtype).tiny,
                )
            )
            for name, count in [("single", 1), ("multi", 2)]
        )

    def gradient(values):
        model = NoiseBlurModel(*values, dtype=dtype)
        result = np.zeros(3, dtype=dtype)
        floor = 1e-300 if dtype.itemsize == 8 else np.finfo(dtype).tiny
        for index, (name, count) in enumerate((("single", 1), ("multi", 2))):
            probability = _bin_probability(model, count)
            ratio = np.divide(
                pooled[name],
                np.maximum(probability, floor),
                out=np.zeros_like(probability),
                where=probability > floor,
            )
            pair = -dtype.type(0.5) * (ratio @ _bin_probability_gradient(model, count))
            result[0] += pair[0]
            result[index + 1] = pair[1]
        return result

    starts = [(0.5, 0.8, 0.95), (0.8, 0.5, 0.9), (0.95, 0.7, 0.95), (0.2, 0.3, 0.7)]
    results = [
        minimize(
            objective,
            start,
            method="L-BFGS-B",
            jac=gradient,
            bounds=[(0.01, 0.99), (0.01, 1.0), (0.01, 1.0)],
            options={
                "ftol": 1e-13,
                "gtol": 1e-7,
                "maxiter": 1000,
            },
        )
        for start in starts
    ]
    successful = [r for r in results if r.success and np.isfinite(r.fun)]
    if not successful:
        raise RuntimeError(
            "No Gaussian-mixture optimization start converged: "
            + str([r.message for r in results])
        )
    best = min(successful, key=lambda r: r.fun)
    diagnostics = {
        "objective": float(best.fun),
        "objective_name": "case-balanced binned negative log likelihood",
        "family_weights": {"single": 0.5, "multi": 0.5},
        "cases": len(cases),
        "samples_used": sum(c.statistics.sample_count for c in cases),
        "starts": [
            {
                "success": bool(r.success),
                "objective": float(r.fun),
                "parameters": r.x.tolist(),
                "message": str(r.message),
            }
            for r in results
        ],
    }
    return NoiseBlurModel(*best.x, dtype=dtype), diagnostics


def assess_noise_blur(model: NoiseBlurModel, case: ReplayCase) -> dict:
    """Compare fitted and empirical CDFs and fourth moment on the same shape scale."""
    measured = case.statistics
    narrow, broad = model.scales(case.ngs_count)
    weight = model.mixture_weight
    projection = weight * ndtr(PROJECTION_EDGES / np.sqrt(narrow)) + (
        1 - weight
    ) * ndtr(PROJECTION_EDGES / np.sqrt(broad))
    radial = model.radial_cdf(RADIAL_EDGES[1:], case.ngs_count)
    fourth = 8 * (weight * narrow**2 + (1 - weight) * broad**2)
    return {
        "case": case.key,
        "ngs": case.ngs_count,
        "population": case.population,
        "samples_used": measured.sample_count,
        "radial_cdf_max_error": float(
            np.max(abs(np.cumsum(measured.radial_probability) - radial))
        ),
        "projection_cdf_max_error": float(
            np.max(abs(measured.projection_cdf - projection))
        ),
        "radial_fourth_error_pct": float(100 * (fourth / measured.radial_fourth - 1)),
    }


def measurement_statistics(
    zero_samples: NDArray,
    shifted_samples: NDArray,
    displacement_mas: float,
    block_draws: int,
    *,
    dtype=None,
) -> dict:
    """Calculate full response/noise matrices from paired angular samples.

    Zero samples are (N0,2); shifted samples are (4,N,2), ordered +x,-x,+y,-y.
    Adjacent signed inputs retain paired draw/block identities. Conditional
    mean removal supplies population noise covariance and whitened zero draws.
    Statistics accumulate in the selected dtype, inferred from inputs if omitted.
    """
    dtype = validate_psf_dtype(
        np.asarray(zero_samples).dtype if dtype is None else dtype
    )
    zero = np.asarray(zero_samples, dtype=dtype)
    shifted = np.asarray(shifted_samples, dtype=dtype)
    displacement_mas = dtype.type(displacement_mas)
    if (
        zero.ndim != 2
        or zero.shape[1] != 2
        or len(zero) < 4
        or shifted.ndim != 3
        or shifted.shape[0] != 4
        or shifted.shape[2] != 2
        or not np.isfinite(zero).all()
        or not np.isfinite(shifted).all()
        or type(block_draws) is not int
        or block_draws < 1
        or shifted.shape[1] % block_draws
        or shifted.shape[1] < 2 * block_draws
        or not np.isfinite(displacement_mas)
        or displacement_mas <= 0
    ):
        raise ValueError("Invalid paired samples, displacement or gain blocks.")
    mean = zero.mean(0)
    centred = zero - mean
    covariance = centred.T @ centred / len(zero)
    white = np.linalg.solve(np.linalg.cholesky(covariance), centred.T).T
    blocks = np.stack(
        [
            (shifted[0] - shifted[1]).reshape(-1, block_draws, 2).mean(1),
            (shifted[2] - shifted[3]).reshape(-1, block_draws, 2).mean(1),
        ],
        axis=-1,
    ) / (2 * displacement_mas)
    gain_matrix = blocks.mean(0)
    means = shifted.mean(1).reshape(2, 2, 2).transpose(2, 0, 1)
    return {
        "mean": mean,
        "covariance": covariance,
        "white": white.astype(dtype),
        "gain_matrix": gain_matrix,
        "gain": np.trace(gain_matrix) / 2,
        "gain_se": blocks.std(0, ddof=1) / np.sqrt(dtype.type(len(blocks))),
        "paired_gain_blocks": blocks,
        "shifted_means": means,
    }
