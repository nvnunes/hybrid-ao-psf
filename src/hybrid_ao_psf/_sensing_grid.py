"""Measured magnitude support and rectangular noise-bank mixing weights."""

import itertools

import numpy as np
from numpy.typing import ArrayLike


def calibration_magnitude_intervals(
    magnitudes: ArrayLike, declared: ArrayLike | None = None
) -> list[list[float]]:
    """Return supported intervals for a sorted measured magnitude axis.

    Explicit intervals permit calibrated nonuniform node spacing. Their ends
    must be measured nodes, intervals must be ordered and disjoint, and every
    node must be covered. Without a declaration, retain the conservative gap
    rule: spacing above the minimum splits support.
    """
    magnitudes = np.asarray(magnitudes)
    if (
        magnitudes.ndim != 1
        or not len(magnitudes)
        or not np.isfinite(magnitudes).all()
        or np.any(np.diff(magnitudes) <= 0)
    ):
        raise ValueError("Measured magnitudes must be finite and increasing.")
    if declared is not None:
        intervals = np.asarray(declared, dtype=magnitudes.dtype)
        if (
            intervals.ndim != 2
            or intervals.shape[1] != 2
            or not len(intervals)
            or not np.isfinite(intervals).all()
            or np.any(intervals[:, 0] > intervals[:, 1])
            or np.any(intervals[1:, 0] <= intervals[:-1, 1])
            or any(
                not np.any(np.isclose(endpoint, magnitudes, rtol=0, atol=1e-10))
                for endpoint in intervals.ravel()
            )
        ):
            raise ValueError(
                "Magnitude intervals must be ordered, disjoint and end at measured nodes."
            )
        covered = np.any(
            (magnitudes[:, None] >= intervals[:, 0] - 1e-10)
            & (magnitudes[:, None] <= intervals[:, 1] + 1e-10),
            axis=1,
        )
        if not covered.all():
            raise ValueError("Magnitude intervals must cover every measured node.")
        return intervals.tolist()
    differences = np.diff(magnitudes)
    max_gap = float(np.min(differences)) if len(differences) else 0.0
    intervals = []
    first = float(magnitudes[0])
    for left, right in itertools.pairwise(magnitudes):
        if right - left > max_gap + 1e-10:
            intervals.append([first, float(left)])
            first = float(right)
    intervals.append([first, float(magnitudes[-1])])
    return intervals


def rectangular_neighbors(point, axes):
    """Return only nonzero-weight vertices; refuse extrapolation."""
    choices = []
    for value, axis in zip(point, axes, strict=True):
        if value < axis[0] or value > axis[-1]:
            raise ValueError("Requested noise shape lies outside the calibration axes.")
        right = int(np.searchsorted(axis, value))
        if right < len(axis) and np.isclose(axis[right], value, rtol=0, atol=1e-10):
            choices.append([(float(axis[right]), 1.0)])
        else:
            left = right - 1
            weight = (value - axis[left]) / (axis[right] - axis[left])
            choices.append(
                [(float(axis[left]), 1 - weight), (float(axis[right]), weight)]
            )
    return [
        (tuple(v for v, _ in parts), float(np.prod([w for _, w in parts])))
        for parts in itertools.product(*choices)
    ]
