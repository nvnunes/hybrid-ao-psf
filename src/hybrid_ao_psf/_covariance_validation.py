"""Numerical assembly checks and bounded same-PH interpolation diagnostics.

Scientific errors are reported, not clipped or used to tune the calibration.
Whole-PH Gram matrices retain a common feature ordering for uncertainty checks.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

import numpy as np
from ao_core.models.control import scalar_filter, settling_cut
from astropy import units as u
from scipy.interpolate import RegularGridInterpolator

from ._calibration_contracts import CovarianceTelemetry
from ._calibration_products import immutable_metadata
from ._covariance_grid import rectangular_mapping
from ._covariance_statistics import (
    FUNCTIONS,
    FilterSettings,
    HistoryArrays,
    PooledCovariances,
    covariance_tables,
    features,
)

if TYPE_CHECKING:
    from ._covariance_calibration import CovarianceFunctions
    from ._reconstructor_calibration import ReconstructorNoiseModel


BOOTSTRAP_DRAWS = 200
BOOTSTRAP_SEED = 20261013


def _cov(left, right):
    left = left - left.mean(-1, keepdims=True)
    right = right - right.mean(-1, keepdims=True)
    return left @ right.T / left.shape[-1]


def assembly_check(source: HistoryArrays, settings: FilterSettings) -> dict:
    """Compare Q, K, D and residual covariance with independent matrix-loop simulation.

    One-, two- and three-direction fixtures include noncommuting reconstruction
    blocks and unequal eigenvalues. These are numerical fixtures, not fitted
    or physically optimized reconstructors. All comparisons use identical PH
    centring and retained frames, including atmospheric/HO cross terms.
    """
    dtype = source.tilt.dtype
    chosen = np.linspace(
        0, len(source.positions) - 1, min(3, len(source.positions)), dtype=int
    )
    source = HistoryArrays(
        source.tilt[chosen],
        source.coma[chosen],
        source.positions[chosen],
    )
    tables = covariance_tables(source, settings)
    grid_feedback = np.asarray(settings.strengths, dtype=dtype)
    b = grid_feedback[[0, -1]]
    basis = np.array([[1.0, 0.3], [0.1, 1.0]], dtype=dtype)
    inverse = np.linalg.inv(basis)
    projections = [np.outer(basis[:, p], inverse[p]) for p in range(2)]
    maximum, absolute = 0.0, 0.0
    for count in range(1, len(chosen) + 1):
        feedback = basis @ np.diag(b) @ inverse
        weights = [
            np.array([[0.2, 0.08], [-0.04, 0.25]], dtype=dtype)
            for _ in range(count - 1)
        ]
        weights.append(feedback - sum(weights, np.zeros((2, 2), dtype=dtype)))
        h = np.linspace(0.2, 0.5, count, dtype=dtype)
        atmospheric = sum(
            (r @ source.tilt[i] for i, r in enumerate(weights)),
            np.zeros_like(source.tilt[0]),
        )
        coma = sum(
            (h[i] * r @ source.coma[i] for i, r in enumerate(weights)),
            np.zeros_like(source.coma[0]),
        )

        def simulate_loop(history, feedback):
            result = np.broadcast_to(
                np.linalg.solve(feedback, history[:, :1]), history.shape
            ).copy()
            for k in range(settings.delay, history.shape[-1]):
                result[:, k] = (
                    result[:, k - 1]
                    - settings.gain * feedback @ result[:, k - settings.delay]
                    + settings.gain * history[:, k - settings.delay]
                )
            return result[:, settings.discard :]

        ua, uc = simulate_loop(atmospheric, feedback), simulate_loop(coma, feedback)
        science = source.tilt[0, :, settings.discard :]
        predicted_a, predicted_c, predicted_ac = [
            np.zeros((2, 2), dtype=dtype) for _ in range(3)
        ]
        predicted_ka, predicted_kc = (
            np.zeros((2, 2), dtype=dtype),
            np.zeros((2, 2), dtype=dtype),
        )
        for i, ri in enumerate(weights):
            for p, pp in enumerate(projections):
                ip = 0 if p == 0 else len(grid_feedback) - 1
                predicted_ka += tables["C_t_Tt"][0, i, ip] @ (pp @ ri).T
                predicted_kc += h[i] * tables["C_t_Tc"][0, i, ip] @ (pp @ ri).T
                for g, rg in enumerate(weights):
                    for q, pq in enumerate(projections):
                        iq = 0 if q == 0 else len(grid_feedback) - 1
                        left, right = pp @ ri, pq @ rg
                        predicted_a += left @ tables["C_Tt_Tt"][i, g, ip, iq] @ right.T
                        predicted_c += (
                            h[i]
                            * h[g]
                            * left
                            @ tables["C_Tc_Tc"][i, g, ip, iq]
                            @ right.T
                        )
                        predicted_ac += (
                            h[g] * left @ tables["C_Tt_Tc"][i, g, ip, iq] @ right.T
                        )
        d = predicted_kc - predicted_ac
        residual = (
            tables["C_t_t"][0, 0]
            + predicted_a
            + predicted_c
            - predicted_ka
            - predicted_ka.T
            - d
            - d.T
        )
        comparisons = [
            (predicted_a, _cov(ua, ua)),
            (predicted_c, _cov(uc, uc)),
            (predicted_ac, _cov(ua, uc)),
            (predicted_ka, _cov(science, ua)),
            (predicted_kc, _cov(science, uc)),
            (d, _cov(science - ua, uc)),
            (residual, _cov(science - ua - uc, science - ua - uc)),
        ]
        for prediction, expected in comparisons:
            error = np.linalg.norm(prediction - expected)
            scale = max(
                np.linalg.norm(prediction),
                np.linalg.norm(expected),
                np.finfo(float).tiny,
            )
            maximum = max(maximum, float(error / scale))
            absolute = max(absolute, float(error))
    # Matrix reconstruction and covariance products accumulate several
    # rounding steps; retain the original double-precision gate.
    if maximum > max(1e-10, 64 * np.finfo(source.tilt.dtype).eps):
        raise ValueError(
            f"Independent matrix-loop covariance assembly failed: {maximum}"
        )
    return {
        "max_scaled_error": maximum,
        "max_absolute_error": absolute,
        "direction_counts": list(range(1, len(chosen) + 1)),
    }


class InterpolationDiagnostics:
    """Accumulate same-PH withheld feedback and reduced-grid spatial checks.

    Feedback checks use midpoints between adjacent values and unequal strength pairs.
    Spatial checks use every other point on the full grid, fill from the nearest
    measured points and compare at withheld directions. They describe
    that coarser grid; they are not off-grid truth for the final interpolator.
    Only a bounded set of direction pairs is retained, not full per-PH covariance tables.
    """

    def __init__(self, source: HistoryArrays, settings: FilterSettings):
        self.dtype = source.tilt.dtype
        self.settings = settings
        self.positions = source.positions
        self.indices = np.linspace(
            0, len(source.positions) - 1, min(3, len(source.positions)), dtype=int
        )
        self.midpoints = (
            np.array(settings.strengths[:-1], dtype=self.dtype) + settings.strengths[1:]
        ) / 2
        self.feedback_predicted = PooledCovariances()
        self.feedback_truth = PooledCovariances()
        self.spatial_predicted = PooledCovariances()
        self.spatial_truth = PooledCovariances()
        self.count = 0
        xs, ys = np.unique(self.positions[:, 0]), np.unique(self.positions[:, 1])
        if min(len(xs), len(ys)) < 3:
            self.training = np.arange(len(self.positions))
            self.withheld = np.array([], dtype=int)
            self.weights = np.empty((0, len(self.positions)), dtype=self.dtype)
            return
        training = np.flatnonzero(
            np.isin(self.positions[:, 0], xs[::2])
            & np.isin(self.positions[:, 1], ys[::2])
        )
        self.training = training
        cx, cy, mapping, _ = rectangular_mapping(self.positions[training])
        weights = RegularGridInterpolator(
            (cx, cy),
            np.eye(len(training), dtype=self.dtype)[mapping],
            bounds_error=True,
        )
        withheld = np.flatnonzero(~np.isin(np.arange(len(self.positions)), training))
        withheld = withheld[
            (self.positions[withheld, 0] >= cx[0])
            & (self.positions[withheld, 0] <= cx[-1])
            & (self.positions[withheld, 1] >= cy[0])
            & (self.positions[withheld, 1] <= cy[-1])
        ]
        self.withheld = (
            withheld[
                np.linspace(0, len(withheld) - 1, min(5, len(withheld)), dtype=int)
            ]
            if len(withheld)
            else withheld
        )
        self.weights = np.asarray(
            weights(self.positions[self.withheld]), dtype=self.dtype
        )

    def add(self, source: HistoryArrays) -> None:
        """Add one independent PH with the common full-grid exposure convention."""
        t, ft, fc = features(source, self.settings)
        small = HistoryArrays(
            source.tilt[self.indices],
            source.coma[self.indices],
            source.positions[self.indices],
        )
        if len(self.midpoints):

            def midpoint_features(history):
                filtered = np.stack(
                    [
                        scalar_filter(
                            history,
                            b,
                            self.settings.gain,
                            self.settings.delay,
                            dtype=None,
                        )[..., self.settings.discard :]
                        for b in self.midpoints
                    ],
                    axis=1,
                )
                return filtered

            mt, mft, mfc = (
                t[self.indices],
                midpoint_features(small.tilt),
                midpoint_features(small.coma),
            )
            b = np.array(self.settings.strengths, dtype=self.dtype)

            def interpolate(history):
                # Covariance interpolation is the same linear map on both inputs.
                normalized = history[self.indices] * b[None, :, None, None]
                return (normalized[:, :-1] + normalized[:, 1:]) / (
                    2 * self.midpoints[None, :, None, None]
                )

            self.feedback_predicted.add((mt, interpolate(ft), interpolate(fc)))
            self.feedback_truth.add((mt, mft, mfc))
        if len(self.withheld):
            selected_b = [0, len(self.settings.strengths) - 1]
            interpolated = [
                np.einsum("qd,d...t->q...t", self.weights, a[self.training])
                for a in (t, ft[:, selected_b], fc[:, selected_b])
            ]
            actual = [
                a[self.withheld] for a in (t, ft[:, selected_b], fc[:, selected_b])
            ]
            self.spatial_predicted.add(tuple(interpolated))
            self.spatial_truth.add(tuple(actual))
        self.count += 1

    def report(self) -> dict:
        """Return absolute and aggregate Frobenius errors without adopting a tolerance."""

        def errors(predicted, truth):
            predicted, truth = predicted.tables(), truth.tables()
            result = {}
            for name in FUNCTIONS:
                expected = truth[name]
                difference = predicted[name] - expected
                norm = max(float(np.linalg.norm(expected)), np.finfo(float).tiny)
                result[name] = {
                    "relative_frobenius": float(np.linalg.norm(difference) / norm),
                    "max_absolute": float(np.max(abs(difference))),
                }
            return result

        if not self.count:
            raise ValueError("No PHs for interpolation diagnostics")
        return {
            "ph_count": self.count,
            "feedback": errors(self.feedback_predicted, self.feedback_truth),
            "midpoint_strengths": self.midpoints.tolist(),
            "feedback_direction_indices": self.indices.tolist(),
            "spatial": errors(self.spatial_predicted, self.spatial_truth)
            if len(self.withheld)
            else None,
            "spatial_training_count": len(self.training),
            "spatial_withheld_indices": self.withheld.tolist(),
            "limitations": (
                "Same-PH checks at withheld points on a coarser spatial grid. "
                "No reference comparison between points on the final grid or between zenith angles."
            ),
        }


def precision_report(
    grams: np.ndarray,
    means: np.ndarray,
    *,
    direction_count: int,
    strength_count: int,
) -> dict:
    """Whole-PH bootstrap precision and joint PSD of the retained feature set.

    grams contains within-PH covariance; means contains each PH's time mean.
    Resample both together and recalculate the pooled mean in each draw.
    Resample PHs, never correlated frames. Report aggregate covariance uncertainty
    and independent half-ensemble differences; do not call them interpolation
    errors. PSD is checked after scaling each variable by its own RMS. Features
    are ordered t(direction,axis), ft(direction,strength,axis), then fc with
    the same filtered layout. Cardinalities explicitly define block boundaries.
    """
    unfiltered, filtered = 2 * direction_count, 2 * direction_count * strength_count
    total = unfiltered + 2 * filtered
    if (
        direction_count < 1
        or strength_count < 1
        or grams.ndim != 3
        or grams.shape[1:] != (total, total)
        or not len(grams)
        or not np.isfinite(grams).all()
        or means.shape != (len(grams), total)
        or not np.isfinite(means).all()
    ):
        raise ValueError("Invalid precision feature layout or Gram samples")

    def pooled(indices):
        selected = means[indices]
        centered = selected - selected.mean(0)
        return grams[indices].mean(0) + centered.T @ centered / len(indices)

    mean = pooled(np.arange(len(grams)))
    rms = np.sqrt(np.diag(mean))
    scale = np.outer(rms, rms)
    correlation = np.divide(mean, scale, out=np.zeros_like(mean), where=scale > 0)
    minimum = float(np.linalg.eigvalsh(correlation).min())
    if minimum < -max(1e-12, 8 * total * np.finfo(grams.dtype).eps):
        raise ValueError(f"Joint feature covariance is not PSD: {minimum}")
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    samples = np.array(
        [
            pooled(rng.integers(0, len(grams), len(grams)))
            for _ in range(BOOTSTRAP_DRAWS)
        ]
    )
    errors = np.linalg.norm(samples - mean, axis=(1, 2)) / max(
        np.linalg.norm(mean), np.finfo(float).tiny
    )
    half = len(grams) // 2
    t = slice(0, unfiltered)
    ft = slice(unfiltered, unfiltered + filtered)
    fc = slice(unfiltered + filtered, total)
    blocks = {
        "C_t_t": (t, t),
        "C_Tc_Tc": (fc, fc),
        "C_t_Tc": (t, fc),
        "C_Tt_Tc": (ft, fc),
        "C_Tt_Tt": (ft, ft),
        "C_t_Tt": (t, ft),
    }
    block_errors = {}
    absolute_errors = {}
    for name, (left, right) in blocks.items():
        norm = max(np.linalg.norm(mean[left, right]), np.finfo(float).tiny)
        errors_block = (
            np.linalg.norm(samples[:, left, right] - mean[left, right], axis=(1, 2))
            / norm
        )
        block_errors[name] = np.percentile(errors_block, [16, 50, 84]).tolist()
        absolute_errors[name] = np.percentile(
            errors_block * norm, [16, 50, 84]
        ).tolist()
    return {
        "bootstrap_samples": BOOTSTRAP_DRAWS,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "ph_count": len(grams),
        "direction_count": direction_count,
        "strength_count": strength_count,
        "relative_frobenius_percentiles_16_50_84": np.percentile(
            errors, [16, 50, 84]
        ).tolist(),
        "function_relative_frobenius_percentiles_16_50_84": block_errors,
        "function_absolute_frobenius_percentiles_16_50_84": absolute_errors,
        "independent_half_relative_frobenius": float(
            np.linalg.norm(
                pooled(np.arange(half)) - pooled(np.arange(half, len(grams)))
            )
            / max(np.linalg.norm(mean), np.finfo(float).tiny)
        )
        if half
        else None,
        "joint_correlation_min_eigenvalue": minimum,
        "limitations": "Precision of selected direction/strength joint covariance; not a guarantee for all queries.",
    }


def joint_query_check(model) -> dict:
    """Check one mixed-field/mixed-feedback joint covariance from all six functions.

    These are interpolation-validity checks, not comparisons with off-grid
    atmospheric truth. No eigenvalue clipping or covariance symmetrization is
    applied to the returned functions.
    """
    names = {
        (0, 0): "C_t_t",
        (1, 1): "C_Tt_Tt",
        (2, 2): "C_Tc_Tc",
        (0, 1): "C_t_Tt",
        (0, 2): "C_t_Tc",
        (1, 2): "C_Tt_Tc",
    }
    queries = [
        (
            model.x[0] + f * (model.x[-1] - model.x[0]),
            model.y[0] + g * (model.y[-1] - model.y[0]),
            model.strengths[0] + h * (model.strengths[-1] - model.strengths[0]),
        )
        for f, g, h in [(0.18, 0.64, 0.1), (0.53, 0.39, 0.45), (0.86, 0.13, 0.9)]
    ]
    matrix = np.empty((18, 18), dtype=model.dtype)
    for i, (x, y, b) in enumerate(queries):
        for j, (xx, yy, bb) in enumerate(queries):
            for (left, right), name in names.items():
                args = [x, y, xx, yy] + ([b] if left else []) + ([bb] if right else [])
                block = getattr(model, name)(*args, zenith_deg=float(model.zeniths[0]))
                row, col = 2 * (3 * i + left), 2 * (3 * j + right)
                matrix[row : row + 2, col : col + 2] = block
                if left != right:
                    matrix[col : col + 2, row : row + 2] = block.T
    symmetry = float(
        np.linalg.norm(matrix - matrix.T)
        / max(np.linalg.norm(matrix), np.finfo(float).tiny)
    )
    rms = np.sqrt(np.diag(matrix))
    scale = np.outer(rms, rms)
    correlation = np.divide(matrix, scale, out=np.zeros_like(matrix), where=scale > 0)
    minimum = float(np.linalg.eigvalsh(correlation).min())
    if symmetry > max(1e-12, 16 * np.finfo(matrix.dtype).eps) or minimum < -max(
        1e-12, 8 * len(matrix) * np.finfo(matrix.dtype).eps
    ):
        raise ValueError(
            f"Interpolated joint covariance invalid: symmetry={symmetry}, eigenvalue={minimum}"
        )
    return {
        "symmetry_scaled_error": symmetry,
        "joint_correlation_min_eigenvalue": minimum,
        "queries": np.array(queries).tolist(),
    }


def pilot_diagnostics(
    sources: list[HistoryArrays],
    settings: FilterSettings,
    *,
    tolerance: float = 1e-8,
    maximum: int = 500,
) -> dict:
    """Check recurrence, initialization and restarted filtering on actual PH inputs.

    A direct frame recurrence supplies the independent scalar reference.
    Covariance-change errors are scaled by variable RMS products so that weak
    cross-covariances are not divided by their own near-zero values.
    """
    cut, envelope = settling_cut(
        settings.gain,
        settings.delay,
        settings.strengths,
        tolerance=tolerance,
        maximum=maximum,
    )
    if settings.discard < cut:
        raise ValueError(
            f"Settling cut {settings.discard} is below the state-propagation bound {cut}"
        )
    errors, settling, assemblies, absolute_settling = [], [], [], []
    for source in sources[:2]:
        assemblies.append(assembly_check(source, settings))
        indices = np.linspace(
            0, len(source.positions) - 1, min(3, len(source.positions)), dtype=int
        )
        for input_name, values in [
            ("tilt", source.tilt[indices]),
            ("coma", source.coma[indices]),
        ]:
            for b in settings.strengths:
                output = scalar_filter(
                    values, b, settings.gain, settings.delay, dtype=None
                )
                expected = np.broadcast_to(values[..., :1] / b, values.shape).copy()
                for k in range(settings.delay, values.shape[-1]):
                    expected[..., k] = (
                        expected[..., k - 1]
                        - settings.gain * b * expected[..., k - settings.delay]
                        + settings.gain * values[..., k - settings.delay]
                    )
                scale = max(float(np.max(abs(expected))), np.finfo(float).tiny)
                error = float(np.max(abs(expected - output)) / scale)
                if error > max(1e-10, 64 * np.finfo(values.dtype).eps):
                    raise ValueError(
                        f"Scalar recurrence disagreement: {input_name}, b={b}, {error}"
                    )
                errors.append(error)
                restart = min(200, source.tilt.shape[-1] // 4)
                if restart + settings.discard >= values.shape[-1] - 3:
                    raise ValueError(
                        "Saved exposure too short for restarted settling comparison"
                    )
                baseline = output[..., restart + settings.discard :].copy()
                baseline -= baseline.mean(-1, keepdims=True)
                for equilibrium in [True, False]:
                    restarted = scalar_filter(
                        values[..., restart:],
                        b,
                        settings.gain,
                        settings.delay,
                        equilibrium=equilibrium,
                        dtype=None,
                    )[..., settings.discard :]
                    restarted -= restarted.mean(-1, keepdims=True)
                    n = restarted.shape[-1]
                    left, right = baseline.reshape(-1, n), restarted.reshape(-1, n)
                    difference = right @ right.T / n - left @ left.T / n
                    rms = np.sqrt(np.mean(left**2, axis=1))
                    denominator = np.outer(rms, rms)
                    normalized = np.divide(
                        abs(difference),
                        denominator,
                        out=np.zeros_like(difference),
                        where=denominator > 0,
                    )
                    value = float(normalized.max())
                    if value > 1e-4:
                        raise ValueError(
                            f"Filter settling failed: {input_name}, b={b}, error={value}"
                        )
                    settling.append(value)
                    absolute_settling.append(float(abs(difference).max()))
    return {
        "state_cut": cut,
        "selected_cut": settings.discard,
        "transition_envelope": envelope,
        "max_recurrence_scaled_error": max(errors),
        "max_restart_covariance_scaled_error": max(settling),
        "max_restart_covariance_absolute_error": max(absolute_settling),
        "matrix_loop_assembly": assemblies,
        "retained_frames": sources[0].tilt.shape[-1] - settings.discard,
    }


# Validate filtered covariance against direct loop simulation


def _check_covariance_against_loop(result, reconstructor, validation_telemetry):
    """Attach a direct-loop comparison for each measured zenith plane."""
    from ._covariance_calibration import CovarianceCalibrationResult

    checks = {}
    for telemetry in validation_telemetry:
        angle = float(telemetry.zenith_angle.value)
        if angle in checks:
            raise ValueError("duplicate validation telemetry planes")
        checks[angle] = _validate_covariance_reconstructor(
            result.model, reconstructor, telemetry
        )
    if set(checks) != set(result.model.zenith_angles.value):
        raise ValueError("validation telemetry must cover every measured ZA plane")
    return CovarianceCalibrationResult(
        model=result.model,
        diagnostics={
            **result.diagnostics,
            "calibrated_reconstructor": {
                str(angle): check for angle, check in checks.items()
            },
        },
    )


def _validate_covariance_reconstructor(
    covariances: CovarianceFunctions,
    noise: ReconstructorNoiseModel,
    telemetry: CovarianceTelemetry,
) -> Mapping:
    """Check calibrated-weight atmospheric assembly against direct matrix simulation.

    Use matching uncentred telemetry for one measured ZA plane and the interpolator's
    settled window. Choose one/two/three measured directions, build absolute
    weights with the calibrated prior, and use a controlled full-matrix sensing
    response whose feedback eigenvalues are grid points. This isolates numerical
    assembly from feedback interpolation and physical gain modelling. Compare
    command, science/command and residual covariances using one pooled mean.
    Raise on incompatible timing/directions or discrepancies above rtol=1e-12,
    atol=1e-10 mas squared. Return immutable per-asterism numerical evidence.
    No native files, WFS model or MASTSEL calculation is used.

    Args:
        covariances: Measured covariance interpolators.
        noise: Accepted reconstructor noise prior.
        telemetry: Matched histories at one measured zenith plane.

    Returns:
        Immutable per-asterism evidence comparing interpolator and direct simulation.

    Raises:
        TypeError: If inputs are not the required typed records.
        ValueError: If compatibility or agreement checks fail.
    """
    from ._covariance_calibration import CovarianceFunctions
    from ._reconstructor_calibration import (
        ReconstructorNoiseModel,
        build_lo_reconstructor,
    )

    if (
        not isinstance(covariances, CovarianceFunctions)
        or not isinstance(noise, ReconstructorNoiseModel)
        or not isinstance(telemetry, CovarianceTelemetry)
    ):
        raise TypeError(
            "calibrated interpolator, prior and physical telemetry required"
        )
    if telemetry.frame_rate != covariances.controller.frame_rate:
        raise ValueError("validation and covariance frame rates differ")
    dtype = covariances.dtype
    cut = covariances.discard
    if telemetry.tilt.shape[1] - cut < 3:
        raise ValueError("insufficient validation frames after settling")
    feedback_endpoints = covariances.feedback[[0, -1]]
    feedback = np.diag(feedback_endpoints)
    chosen = np.linspace(
        0, len(telemetry.positions) - 1, min(3, len(telemetry.positions)), dtype=int
    )
    science_position = telemetry.positions[chosen[0]]
    science = telemetry.tilt.value[:, cut:, chosen[0]]
    width = np.sqrt(noise.fwhm.value[0] * noise.fwhm.value[-1])
    signal = np.sqrt(
        noise.photons_per_frame.value[0] * noise.photons_per_frame.value[-1]
    )
    fraction = (
        0.5 if noise.ee is None else np.sqrt(noise.ee.value[0] * noise.ee.value[-1])
    )
    projections = [np.diag([1, 0]), np.diag([0, 1])]
    angle = telemetry.zenith_angle
    evidence = {}

    def covariance(left, right):
        left, right = left.reshape(-1, 2), right.reshape(-1, 2)
        return (left - left.mean(0)).T @ (right - right.mean(0)) / len(left)

    for count in range(1, len(chosen) + 1):
        indices = chosen[:count]
        positions = telemetry.positions[indices]
        photons = signal / fraction if noise.ee is None else signal
        weights = build_lo_reconstructor(
            covariances,
            noise,
            ngs_positions=positions,
            zenith_angle=angle,
            photons_per_frame=np.full(count, photons, dtype=dtype) * u.photon,
            ee=np.full(count, fraction, dtype=dtype) * u.one,
            fwhm=np.full(count, width, dtype=dtype) * u.mas,
            dtype=dtype,
        )
        blocks = [weights[:, 2 * g : 2 * g + 2] for g in range(count)]
        sensing = np.linalg.solve(sum(blocks, np.zeros((2, 2), dtype=dtype)), feedback)
        transforms = [block @ sensing for block in blocks]
        inputs = sum(
            (
                telemetry.tilt.value[:, :, index] @ transform.T
                for index, transform in zip(indices, transforms, strict=True)
            ),
            np.zeros(telemetry.tilt.shape[:2] + (2,), dtype=dtype),
        )
        commands = np.broadcast_to(
            np.linalg.solve(feedback, inputs[:, 0].T).T[:, None], inputs.shape
        ).copy()
        gain, delay = covariances.controller.gain, covariances.controller.delay
        for frame in range(delay, inputs.shape[1]):
            commands[:, frame] = (
                commands[:, frame - 1]
                - gain * (commands[:, frame - delay] @ feedback.T)
                + gain * inputs[:, frame - delay]
            )
        commands = commands[:, cut:]
        q, k = np.zeros((2, 2), dtype=dtype), np.zeros((2, 2), dtype=dtype)
        for (xi, yi), left in zip(positions, transforms, strict=True):
            for p, projection in enumerate(projections):
                a = projection @ left
                k += (
                    covariances.C_t_Tt(
                        *science_position,
                        xi,
                        yi,
                        feedback_endpoints[p],
                        zenith_angle=angle,
                    ).value
                    @ a.T
                )
                for (xg, yg), right in zip(positions, transforms, strict=True):
                    for j, other in enumerate(projections):
                        b = other @ right
                        q += (
                            a
                            @ covariances.C_Tt_Tt(
                                xi,
                                yi,
                                xg,
                                yg,
                                feedback_endpoints[p],
                                feedback_endpoints[j],
                                zenith_angle=angle,
                            ).value
                            @ b.T
                        )
        residual = (
            covariances.C_t_t(
                *science_position, *science_position, zenith_angle=angle
            ).value
            + q
            - k
            - k.T
        )
        comparisons = {
            "command": (q, covariance(commands, commands)),
            "science_command": (k, covariance(science, commands)),
            "residual": (residual, covariance(science - commands, science - commands)),
        }
        errors = {}
        for name, (predicted, expected) in comparisons.items():
            roundoff = (
                128
                * np.finfo(dtype).eps
                * max(np.linalg.norm(predicted), np.linalg.norm(expected))
            )
            if not np.allclose(
                predicted, expected, rtol=1e-12, atol=max(1e-10, roundoff)
            ):
                raise ValueError(
                    f"calibrated-reconstructor validation failed: {count} NGS, {name}"
                )
            errors[name] = float(np.max(np.abs(predicted - expected)))
        evidence[str(count)] = {
            "weights": weights.tolist(),
            "max_absolute_mas2": errors,
        }
    return immutable_metadata(evidence)
