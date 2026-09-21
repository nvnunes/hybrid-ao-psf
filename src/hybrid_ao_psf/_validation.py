"""Validation helpers shared by Hybrid AO PSF public contracts."""

from __future__ import annotations

import re
from typing import Any

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
