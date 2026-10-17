"""Response regression, paired diagnostics and the three-parameter h law.

The estimators, wavefront bootstrap, Nelder-Mead conditioning and acceptance
criteria follow the retained calibration. All public responses use mas/nm OPD.
"""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize


@dataclass(frozen=True)
class _ResponseArrays:
    coma: np.ndarray
    magnitudes: np.ndarray
    tilt_mean: np.ndarray
    tilt_batches: np.ndarray
    tilt_noiseless: np.ndarray


def _regression(coma, tilt):
    dtype = np.asarray(coma).dtype
    coma = coma - coma.mean(axis=0)
    tilt = tilt - tilt.mean(axis=0)
    denominator = np.sum(coma * coma)
    if denominator <= 0:
        raise ValueError("Coma measurements must vary across wavefronts")
    h = np.sum(coma * tilt) / denominator
    gram = coma.T @ coma
    # MATLAB rcond uses the one-norm condition estimate.
    if 1 / np.linalg.cond(gram, 1) > np.finfo(dtype).eps:
        matrix = np.linalg.solve(gram.T, (tilt.T @ coma).T).T
    else:
        matrix = np.full((2, 2), np.nan, dtype=dtype)
    return h, matrix


def _resample(rng, coma):
    # MATLAB rng(seed,'twister'); randi(N,1,N) uses these uniform draws.
    # Keep its redraw-only-zero-variance rule and the full/half stream order.
    while True:
        ids = np.floor(rng.random_sample(len(coma)) * len(coma)).astype(int)
        if np.any(coma[ids] != coma[ids[0]]):
            return ids


def _simplex(objective, initial, *, joint=False):
    """Use the fminsearch simplex, coefficients, stopping rules and budgets."""
    initial = np.asarray(initial)
    # A float32 simplex cannot shrink below representable coordinate steps.
    xatol = max(
        1e-10, 4 * np.finfo(initial.dtype).eps * max(1.0, float(np.max(abs(initial))))
    )
    # The float32 callback cost also has finite resolution. Keep the retained
    # double-precision stopping rules exactly as specified.
    fatol = (
        1e-12
        if initial.dtype == np.dtype("float64")
        else max(
            1e-12,
            8 * np.finfo(initial.dtype).eps * max(1.0, float(abs(objective(initial)))),
        )
    )
    return minimize(
        objective,
        initial,
        method="Nelder-Mead",
        options={
            "xatol": xatol,
            "fatol": fatol,
            "maxiter": 3000 if joint else 200 * len(initial),
            "maxfev": 6000 if joint else 200 * len(initial),
        },
    )


def response_law(parameters, airmass, magnitudes):
    """Evaluate the joint law in mas/nm for (ZA,magnitude) broadcast grids."""
    dtype = np.asarray(magnitudes).dtype
    a0, b, half = np.asarray(parameters, dtype=dtype)
    return (a0 + b * (np.asarray(airmass).reshape(-1, 1) - 1)) / (
        1 + 10 ** (0.4 * (np.asarray(magnitudes) - half))
    )


def fit_response(
    response: _ResponseArrays,
    *,
    sensing_wavelength: float,
    bootstrap_seed: int,
    noise_draws: int,
) -> dict:
    """Estimate one ZA with the retained 200-replicate wavefront bootstrap.

    sensing_wavelength is in metres and supplies the existing optimizer's
    numerical conditioning only. bootstrap_seed is the seed previously passed
    to MATLAB rng (PhaseSeed+1). Input means are removed across wavefronts.
    No native WFS objects, physical configuration or acquisition files are read.
    """
    dtype = np.asarray(response.coma).dtype
    if not np.isfinite(sensing_wavelength) or sensing_wavelength <= 0:
        raise ValueError("sensing_wavelength must be positive metres")
    if type(bootstrap_seed) is not int or not 0 <= bootstrap_seed < 2**32:
        raise ValueError("bootstrap_seed must be a uint32-range integer")
    c, t, magnitudes = response.coma, response.tilt_mean, response.magnitudes
    if not np.any(c != c[0]):
        raise ValueError("Coma measurements must vary across wavefronts")
    m = len(magnitudes)
    h = np.empty(m, dtype=dtype)
    matrix = np.empty((m, 2, 2), dtype=dtype)
    for j in range(m):
        h[j], matrix[j] = _regression(c, t[:, j])
    rng = np.random.RandomState(5489 if bootstrap_seed == 0 else bootstrap_seed)
    bootstrap = np.empty((200, m), dtype=dtype)
    matrix_bootstrap = np.empty((200, m, 2, 2), dtype=dtype)
    for b in range(200):
        ids = _resample(rng, c)
        for j in range(m):
            bootstrap[b, j], matrix_bootstrap[b, j] = _regression(c[ids], t[ids, j])
    standard_error = bootstrap.std(axis=0, ddof=1)
    half_h = np.full((2, m), np.nan, dtype=dtype)
    half_error = np.full((2, m), np.nan, dtype=dtype)
    for half, ids in enumerate(np.array_split(np.arange(len(c)), 2)):
        if not np.any(c[ids] != c[ids[0]]):
            continue
        for j in range(m):
            half_h[half, j] = _regression(c[ids], t[ids, j])[0]
        samples = np.empty((200, m), dtype=dtype)
        for b in range(200):
            selected = ids[_resample(rng, c[ids])]
            for j in range(m):
                samples[b, j] = _regression(c[selected], t[selected, j])[0]
        half_error[half] = samples.std(axis=0, ddof=1)

    scale = sensing_wavelength * 1e9 / (2 * np.pi)
    curve = lambda p: p[0] / (1 + 10 ** (0.4 * (magnitudes - p[1])))
    fit = _simplex(
        lambda q: np.sum(
            (curve([np.exp(dtype.type(q[0])), dtype.type(q[1])]) - h * scale) ** 2
        ),
        [np.log(max(h) * scale), np.median(magnitudes)],
    )
    parameters = np.array([np.exp(fit.x[0]) / scale, fit.x[1]], dtype=dtype)
    predicted = curve(parameters)
    difference = matrix - h[:, None, None] * np.eye(2, dtype=dtype)
    matrix_error = np.full_like(matrix, np.nan)
    axis_error = np.full(m, np.nan, dtype=dtype)
    valid_counts = np.zeros(m, dtype=int)
    for j in range(m):
        valid = np.isfinite(matrix_bootstrap[:, j]).all(axis=(1, 2))
        valid_counts[j] = valid.sum()
        if valid.sum() >= 2:
            sampled = matrix_bootstrap[valid, j]
            delta = sampled - bootstrap[valid, j, None, None] * np.eye(2, dtype=dtype)
            matrix_error[j] = delta.std(axis=0, ddof=1)
            axis_error[j] = (sampled[:, 0, 0] - sampled[:, 1, 1]).std(ddof=1)
    batch_count = response.tilt_batches.shape[2]
    if (
        type(noise_draws) is not int
        or noise_draws < batch_count
        or noise_draws % batch_count
    ):
        raise ValueError("noise_draws must split equally between detector batches")
    batch_h = np.array(
        [
            [_regression(c, response.tilt_batches[:, j, b])[0] for j in range(m)]
            for b in range(batch_count)
        ],
        dtype=dtype,
    )
    batch_error = batch_h.std(axis=0, ddof=1) / np.sqrt(dtype.type(batch_count))
    return {
        "h": h,
        "jacobian": matrix,
        "standardError": standard_error,
        "halves": half_h,
        "parameters": parameters,
        "predicted": predicted,
        "rmsError": np.sqrt(np.mean((predicted - h) ** 2)),
        "maxRelativeError": np.max(np.abs(predicted / h - 1)),
        "accepted": bool(
            fit.success
            and np.all(h > 0)
            and magnitudes[0] < parameters[1] < magnitudes[-1]
        ),
        "units": "mas/nm OPD",
        "comaUnits": "nm OPD",
        "sensingWavelength": sensing_wavelength,
        "diagnostics": {
            "scalarResponse": {
                "matrixDifference": difference,
                "matrixDifferenceStandardError": matrix_error,
                "matrixRelativeError": np.linalg.norm(difference, axis=(1, 2))
                / np.linalg.norm(matrix, axis=(1, 2)),
                "axisDifference": matrix[:, 0, 0] - matrix[:, 1, 1],
                "axisDifferenceStandardError": axis_error,
                "validMatrixBootstrapReplicates": valid_counts,
            },
            "wavefrontHalves": {
                "h": half_h,
                "standardError": half_error,
                "difference": half_h[0] - half_h[1],
                "differenceStandardError": np.sqrt(np.sum(half_error**2, axis=0)),
                "relativeDifference": np.abs(half_h[0] - half_h[1]) / np.abs(h),
                "wavefrontsPerHalf": len(c) // 2,
            },
            "noiseAveraging": {
                "available": True,
                "hByBatch": batch_h,
                "standardErrorOfMean": batch_error,
                "relativeStandardErrorOfMean": batch_error / np.abs(h),
                "numberOfBatches": batch_count,
                "drawsPerBatch": noise_draws // batch_count,
            },
            "bootstrapReplicates": 200,
        },
    }


def fit_airmass(
    zeniths,
    magnitudes,
    measurements,
    standard_error,
    *,
    sensing_wavelength,
    heldout_zenith=30.0,
):
    """Joint equal-ZA/magnitude fit, retaining the two-SE and held-out checks.

    The caller selects ZA planes and the held-out angle. measurements and
    standard_error are (ZA,magnitude). All three parameters are fitted together.
    """
    dtype = np.asarray(measurements).dtype
    zeniths, magnitudes, measured, error = (
        np.asarray(x, dtype=dtype)
        for x in (zeniths, magnitudes, measurements, standard_error)
    )
    if (
        zeniths.ndim != 1
        or len(zeniths) < 3
        or len(np.unique(zeniths)) != len(zeniths)
        or np.any((zeniths < 0) | (zeniths >= 90))
        or magnitudes.ndim != 1
        or len(magnitudes) < 2
        or np.any(np.diff(magnitudes) <= 0)
        or measured.shape != (len(zeniths), len(magnitudes))
        or error.shape != measured.shape
        or not all(np.isfinite(a).all() for a in (zeniths, magnitudes, measured, error))
        or np.any(measured <= 0)
        or np.any(error < 0)
        or not np.isfinite(sensing_wavelength)
        or sensing_wavelength <= 0
    ):
        raise ValueError("Invalid joint coma-response inputs")
    x = 1 / np.cos(np.deg2rad(zeniths))
    scale = sensing_wavelength * 1e9 / (2 * np.pi)

    def solve(airmass, target):
        target = target * scale
        amplitude = np.linalg.lstsq(
            np.column_stack((np.ones(len(airmass), dtype=dtype), airmass - 1)),
            target[:, 0],
            rcond=None,
        )[0]
        half = np.argmin(np.abs(np.mean(target / target[:, :1], axis=0) - 0.5))
        fit = _simplex(
            lambda p: np.sum((response_law(p, airmass, magnitudes) - target) ** 2),
            [*amplitude, magnitudes[half]],
            joint=True,
        )
        parameters = fit.x.astype(dtype)
        parameters[:2] /= scale
        return parameters, fit.success

    parameters, converged = solve(x, measured)
    predicted = response_law(parameters, x, magnitudes)
    excess = np.max(np.maximum(0, np.abs(predicted - measured) - 2 * error) / measured)
    accepted = bool(
        converged
        and np.all(parameters[0] + parameters[1] * (x - 1) > 0)
        and magnitudes[0] < parameters[2] < magnitudes[-1]
        and excess < 0.1
    )
    result = {
        "parameters": parameters,
        "measured": measured,
        "standardError": error,
        "predicted": predicted,
        "zenithAngles": zeniths,
        "airmass": x,
        "magnitudes": magnitudes,
        "rmsError": np.sqrt(np.mean((predicted - measured) ** 2)),
        "maxRelativeError": np.max(np.abs(predicted / measured - 1)),
        "maxRelativeErrorBeyondSamplingUncertainty": excess,
    }
    heldout = np.isclose(zeniths, heldout_zenith, rtol=0, atol=1e-9)
    if heldout.any() and (~heldout).sum() >= 3:
        p, ok = solve(x[~heldout], measured[~heldout])
        test = response_law(p, x[heldout], magnitudes)
        delta = test - measured[heldout]
        relative = np.max(np.abs(delta / measured[heldout]))
        result["heldout"] = {
            "zenith": heldout_zenith,
            "parameters": p,
            "converged": bool(ok),
            "predicted": test,
            "rmsError": np.sqrt(np.mean(delta**2)),
            "maxRelativeError": relative,
        }
        accepted = accepted and bool(ok and relative < 0.1)
    result["accepted"] = accepted
    return result
