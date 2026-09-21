"""Public interpolation strategy configuration contracts."""

from __future__ import annotations

from dataclasses import dataclass

DEFAULT_RBF_KERNEL = "thin_plate_spline"
DEFAULT_RBF_SMOOTHING = 0.05
DEFAULT_RBF_DEGREE = 1
DEFAULT_REGULAR_GRID_METHOD = "linear"


@dataclass(frozen=True)
class RbfInterpolationConfig:
    """Configuration for scaled SciPy RBF interpolation.

    Attributes:
        kernel: Kernel passed to ``scipy.interpolate.RBFInterpolator``.
        smoothing: Non-negative RBF smoothing value.
        degree: Polynomial degree passed to the SciPy interpolator.
    """

    kernel: str = DEFAULT_RBF_KERNEL
    smoothing: float = DEFAULT_RBF_SMOOTHING
    degree: int = DEFAULT_RBF_DEGREE


@dataclass(frozen=True)
class RegularGridInterpolationConfig:
    """Configuration for persisted regular-grid interpolation.

    Attributes:
        method: Interpolation method. Hybrid AO PSF supports ``linear`` and
            ``nearest``.
    """

    method: str = DEFAULT_REGULAR_GRID_METHOD


__all__ = ["RbfInterpolationConfig", "RegularGridInterpolationConfig"]
