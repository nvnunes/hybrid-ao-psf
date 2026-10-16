"""Private scalar-filtered moments and their exact pooled estimator conventions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ._calibration_numerics import scalar_filter, state_matrix

FUNCTIONS = {
    "C_t_t": (False, False, "mas2"),
    "C_Tc_Tc": (True, True, "nm2"),
    "C_t_Tc": (False, True, "mas nm"),
    "C_Tt_Tc": (True, True, "mas nm"),
    "C_Tt_Tt": (True, True, "mas2"),
    "C_t_Tt": (False, True, "mas2"),
}
NORMALIZATION = {
    name: "b1*b2" if left else "b2" if right else "none"
    for name, (left, right, _) in FUNCTIONS.items()
}


@dataclass(frozen=True)
class FilterSettings:
    """The declared fixed integrator and common PH covariance window.

    Feedback strengths must be positive with stable scalar transitions.
    Discard is a saved-frame count, shared by all six functions.
    """

    strengths: tuple[float, ...] = tuple(np.arange(2, 21) / 10)
    gain: float = 0.3
    delay: int = 2
    discard: int = 285

    def __post_init__(self):
        b = np.asarray(self.strengths, float)
        if (
            b.ndim != 1
            or len(b) < 2
            or not np.isfinite(b).all()
            or np.any(b <= 0)
            or np.any(np.diff(b) <= 0)
        ):
            raise ValueError("Feedback nodes must be increasing, finite and positive")
        if (
            not np.isfinite(self.gain)
            or self.gain <= 0
            or not isinstance(self.delay, int)
            or self.delay < 1
            or not isinstance(self.discard, int)
            or self.discard < 0
        ):
            raise ValueError("Invalid integrator gain, delay or saved-frame cut")
        if any(
            max(abs(np.linalg.eigvals(state_matrix(self.gain, self.delay, value)))) >= 1
            for value in b
        ):
            raise ValueError("Unstable scalar feedback response")

    def record(self) -> dict:
        return {
            "strengths": list(self.strengths),
            "gain": self.gain,
            "delay": self.delay,
            "discard": self.discard,
            "initialization": "first-input equilibrium",
            "centring": "pooled PH/time mean",
            "covariance_divisor": "retained frame count",
            "ph_weight": "equal",
        }


@dataclass(frozen=True)
class HistoryArrays:
    """Frame-last numerical view; units are supplied by the calling calibration.

    tilt and coma have shape (directions, 2, frames); positions are Cartesian
    field coordinates. This view carries no simulator configuration or timing.
    """

    tilt: np.ndarray
    coma: np.ndarray
    positions: np.ndarray


def features(
    telemetry: HistoryArrays, settings: FilterSettings
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return uncentred t, filtered t and filtered c on one common PH window.

    Filter each PH independently. Preserve its mean so pooled covariance keeps
    differences between the mean displacements of separate realizations.
    """
    if telemetry.tilt.shape[-1] - settings.discard < 3:
        raise ValueError("Insufficient retained frames after replay settling")
    ft = np.stack(
        [
            scalar_filter(telemetry.tilt, b, settings.gain, settings.delay)
            for b in settings.strengths
        ],
        axis=1,
    )
    fc = np.stack(
        [
            scalar_filter(telemetry.coma, b, settings.gain, settings.delay)
            for b in settings.strengths
        ],
        axis=1,
    )
    return tuple(a[..., settings.discard :] for a in (telemetry.tilt, ft, fc))


def _cross(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Covariance block with native direction/feedback/axis ordering."""
    count = left.shape[-1]
    raw = (left.reshape(-1, count) @ right.reshape(-1, count).T / count).reshape(
        *left.shape[:-1], *right.shape[:-1]
    )
    if left.ndim == 3 and right.ndim == 3:
        return raw.transpose(0, 2, 1, 3).copy()
    if left.ndim == 3:
        return raw.transpose(0, 2, 3, 1, 4).copy()
    return raw.transpose(0, 3, 1, 4, 2, 5).copy()


def _pairs(values):
    t, ft, fc = values
    return [
        ("C_t_t", t, t),
        ("C_Tc_Tc", fc, fc),
        ("C_t_Tc", t, fc),
        ("C_Tt_Tc", ft, fc),
        ("C_Tt_Tt", ft, ft),
        ("C_t_Tt", t, ft),
    ]


def _within_tables(values):
    centered = tuple(a - a.mean(-1, keepdims=True) for a in values)
    return {name: _cross(left, right) for name, left, right in _pairs(centered)}


def covariance_tables(
    telemetry: HistoryArrays, settings: FilterSettings
) -> dict[str, np.ndarray]:
    """Measure one-PH population covariances for independent assembly checks.

    A single PH has one overall time mean. Use PooledCovariances to combine
    several PHs without removing their different mean displacements.
    """
    return _within_tables(features(telemetry, settings))


class PooledCovariances:
    """Accumulate all six blocks about one common mean across PHs and frames.

    add accepts uncentred (t, filtered t, filtered c) histories with shapes
    (directions,2,frames) and (directions,strengths,2,frames). All PHs must use
    the same shapes and retained window. Only the running within-PH tables
    and each PH's feature means are retained. tables returns new arrays with
    equal PH weights and population divisors, including between-PH covariance.
    """

    def __init__(self):
        self.count = 0
        self._shapes = None
        self._within = None
        self._means = []

    def add(self, values: tuple[np.ndarray, np.ndarray, np.ndarray]) -> None:
        """Add one PH, preserving its time means and cross-covariance blocks."""
        t, ft, fc = values
        shapes = tuple(a.shape for a in values)
        if (
            t.ndim != 3
            or ft.ndim != 4
            or fc.shape != ft.shape
            or t.shape[1] != 2
            or ft.shape[2] != 2
            or t.shape[0] != ft.shape[0]
            or t.shape[-1] != ft.shape[-1]
            or t.shape[0] < 1
            or ft.shape[1] < 1
            or t.shape[-1] < 3
            or any(not np.isfinite(a).all() for a in values)
            or (self._shapes is not None and shapes != self._shapes)
        ):
            raise ValueError("Inconsistent pooled covariance feature histories")
        within = _within_tables(values)
        if self._within is None:
            self._within = within
            self._shapes = shapes
        else:
            for name in FUNCTIONS:
                self._within[name] += within[name]
        self._means.append(tuple(a.mean(-1) for a in values))
        self.count += 1

    def tables(self) -> dict[str, np.ndarray]:
        """Return average within-PH covariance plus covariance of PH means."""
        if not self.count:
            raise ValueError("No PH covariance samples")
        means = tuple(np.stack([m[i] for m in self._means], axis=-1) for i in range(3))
        centered = tuple(a - a.mean(-1, keepdims=True) for a in means)
        return {
            name: self._within[name] / self.count + _cross(left, right)
            for name, left, right in _pairs(centered)
        }
