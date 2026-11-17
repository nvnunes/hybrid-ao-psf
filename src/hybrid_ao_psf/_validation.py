"""Validation helpers shared by Hybrid AO PSF public contracts."""

from __future__ import annotations

import numpy as np
from numpy.typing import DTypeLike


def validate_psf_dtype(dtype: DTypeLike) -> np.dtype:
    """Return one supported PSF pixel dtype without silently changing precision."""
    if dtype is None:
        raise ValueError("PSF dtype must be float32 or float64.")
    try:
        dtype = np.dtype(dtype)
    except (TypeError, ValueError) as error:
        raise ValueError("PSF dtype must be float32 or float64.") from error
    if dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
        raise ValueError("PSF dtype must be float32 or float64.")
    return dtype


def coarsest_precision(*precisions):
    """Coarsest coordinate rounding retained through preparation and loading."""
    return min(
        (validate_psf_dtype(value) for value in precisions if value is not None),
        key=lambda value: value.itemsize,
    )


def coordinate_precision(dtype, *coordinates):
    """Retain float32 source-axis rounding; integer coordinates remain exact."""
    return coarsest_precision(
        dtype,
        *(
            np.dtype("float32")
            for value in coordinates
            if value is not None and value.dtype == np.dtype("float32")
        ),
    )


def snap_coordinate_range(value, axis, *, dtype, label):
    """Snap coordinate roundoff to axis sample points, rejecting extrapolation."""
    axis = np.asarray(axis)
    epsilon = 2 * np.finfo(dtype).eps
    lower = epsilon * max(1.0, abs(float(axis[0])))
    upper = epsilon * max(1.0, abs(float(axis[-1])))
    if np.any(value < axis[0] - lower) or np.any(value > axis[-1] + upper):
        raise ValueError(f"requested {label} coordinates are outside the calibrated range")
    result = np.clip(value, axis[0], axis[-1])
    nearest = np.argmin(np.abs(np.asarray(result)[..., None] - axis), axis=-1)
    # Sample-point roundoff scales locally, including axes spanning orders of magnitude.
    tolerance = epsilon * np.maximum(1.0, np.abs(axis[nearest]))
    return np.where(np.abs(result - axis[nearest]) <= tolerance, axis[nearest], result)
