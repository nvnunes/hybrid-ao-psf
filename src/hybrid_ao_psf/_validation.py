"""Validation helpers shared by Hybrid AO PSF public contracts."""

from __future__ import annotations

import re
from typing import Any

import numpy as np
from numpy.typing import DTypeLike

_DIRECT_FIELD_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_PSF_METADATA_NAMES = frozenset(
    {"wavelength", "pixel_scale", "tel_diameter", "tel_pupil"}
)


def validate_meta_field_name(name: Any, *, label: str = "meta field") -> str:
    """Return one valid provider-owned scalar PSF metadata field name."""
    if not isinstance(name, str):
        raise TypeError(f"{label} names must be strings.")
    normalized = name.strip()
    if not normalized:
        raise ValueError(f"{label} names must be non-empty.")
    if "/" in normalized or not _DIRECT_FIELD_NAME_RE.match(normalized):
        raise ValueError(
            f"{label} name {normalized!r} must start with a letter or underscore "
            "and contain only letters, digits, and underscores."
        )
    if normalized in _PSF_METADATA_NAMES:
        raise ValueError(
            f"{label} name {normalized!r} collides with a core /meta field."
        )
    return normalized


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


def support_dtype(*precisions):
    """Coarsest coordinate rounding retained through preparation and loading."""
    return min(
        (validate_psf_dtype(value) for value in precisions if value is not None),
        key=lambda value: value.itemsize,
    )


def coordinate_support_dtype(dtype, *coordinates):
    """Retain float32 source-axis rounding; integer coordinates remain exact."""
    return support_dtype(
        dtype,
        *(
            np.dtype("float32")
            for value in coordinates
            if value is not None and value.dtype == np.dtype("float32")
        ),
    )


def clip_support(value, axis, *, dtype, label):
    """Snap representational endpoint roundoff, rejecting extrapolation."""
    axis = np.asarray(axis)
    bound = (
        2 * np.finfo(dtype).eps * max(1.0, abs(float(axis[0])), abs(float(axis[-1])))
    )
    if np.any(value < axis[0] - bound) or np.any(value > axis[-1] + bound):
        raise ValueError(f"query outside calibrated {label} support")
    result = np.clip(value, axis[0], axis[-1])
    nearest = np.argmin(np.abs(np.asarray(result)[..., None] - axis), axis=-1)
    return np.where(np.abs(result - axis[nearest]) <= bound, axis[nearest], result)
