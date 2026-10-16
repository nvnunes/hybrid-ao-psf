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
    try:
        dtype = np.dtype(dtype)
    except (TypeError, ValueError) as error:
        raise ValueError("PSF dtype must be float32 or float64.") from error
    if dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
        raise ValueError("PSF dtype must be float32 or float64.")
    return dtype
