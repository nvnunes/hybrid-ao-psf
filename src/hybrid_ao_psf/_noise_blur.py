"""Covariance-preserving two-Gaussian noise blur for Hybrid PSF construction.

The fitted mixture changes distribution shape while preserving a supplied
two-by-two angular covariance. Evaluation requires only the three coefficients,
with no calibration files, E2E software or controller implementation.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
from astropy import units as u
from numpy.typing import ArrayLike, NDArray

__all__ = ["NoiseBlurModel"]


@dataclass(frozen=True)
class NoiseBlurModel:
    """Two centred Gaussian components with an externally supplied covariance.

    Attributes:
        mixture_weight: Weight of the narrow component, strictly between 0 and 1.
        mixture_single_scale: Narrow covariance multiplier for one NGS, in (0,1].
        mixture_multi_scale: Narrow covariance multiplier for multiple NGS, in (0,1].

    The broad scale is chosen so the weighted scales sum to one. Nonfinite or
    out-of-range coefficients raise ValueError at construction.
    """

    mixture_weight: float
    mixture_single_scale: float
    mixture_multi_scale: float

    def __post_init__(self) -> None:
        values = np.array(
            [self.mixture_weight, self.mixture_single_scale, self.mixture_multi_scale]
        )
        if (
            values.shape != (3,)
            or values.dtype.kind not in "fiu"
            or not np.isfinite(values).all()
            or not 0 < values[0] < 1
            or not np.all((values[1:] > 0) & (values[1:] <= 1))
        ):
            raise ValueError(
                "Require weight in (0,1) and narrow covariance scales in (0,1]."
            )
        for name, value in zip(
            ("mixture_weight", "mixture_single_scale", "mixture_multi_scale"),
            values,
            strict=True,
        ):
            object.__setattr__(self, name, float(value))

    def parameters(self) -> dict[str, float]:
        """Return the three named coefficients as ordinary floats."""
        return {
            name: float(getattr(self, name))
            for name in (
                "mixture_weight",
                "mixture_single_scale",
                "mixture_multi_scale",
            )
        }

    def scales(self, ngs_count: int) -> tuple[float, float]:
        """Return narrow/broad covariance multipliers for a positive NGS count.

        One NGS uses the single scale; every larger count uses the multi scale.
        Invalid counts raise ValueError.
        """
        if (
            isinstance(ngs_count, (bool, np.bool_))
            or not isinstance(ngs_count, (int, np.integer))
            or ngs_count < 1
        ):
            raise ValueError("NGS count must be a positive integer.")
        narrow = (
            self.mixture_single_scale if ngs_count == 1 else self.mixture_multi_scale
        )
        broad = (1 - self.mixture_weight * narrow) / (1 - self.mixture_weight)
        return float(narrow), float(broad)

    def radial_cdf(self, squared_radius: ArrayLike, ngs_count: int) -> NDArray:
        """Return the CDF of squared Mahalanobis radius in two dimensions.

        Radius is dimensionless after whitening by the supplied total covariance.
        Scalar/array nonnegative radii, including infinity, are accepted. NaN,
        negative radii and invalid NGS counts raise ValueError.
        """
        radius = np.asarray(squared_radius, float)
        if np.isnan(radius).any() or np.any(radius < 0):
            raise ValueError("Squared radius must be nonnegative.")
        narrow, broad = self.scales(ngs_count)
        weight = self.mixture_weight
        return weight * -np.expm1(-radius / (2 * narrow)) + (1 - weight) * -np.expm1(
            -radius / (2 * broad)
        )

    def otf(
        self, covariance: u.Quantity, ngs_count: int, side: int, pixel_scale: u.Quantity
    ) -> NDArray:
        """Return the normalized OTF on an unshifted square FFT grid.

        ``covariance`` is a finite two-by-two PSD angular-squared quantity;
        its symmetric part is used. ``side`` is the positive integer image
        width/height and ``pixel_scale`` is its positive angular pixel scale.
        Rows/columns follow y/x. The real (side,side) result has DC value one.
        Inputs are not mutated; invalid inputs raise ValueError.
        """
        self.scales(ngs_count)
        if not isinstance(covariance, u.Quantity) or not isinstance(
            pixel_scale, u.Quantity
        ):
            raise TypeError("covariance and pixel_scale must be Astropy quantities")
        return _gaussian_mixture_otf(
            covariance.to_value(u.mas**2),
            self.parameters(),
            ngs_count,
            side,
            pixel_scale.to_value(u.mas),
        )


def _gaussian_mixture_otf(
    covariance: ArrayLike,
    parameters: Mapping[str, float],
    ngs_count: int,
    side: int,
    pixel_mas: float,
) -> NDArray:
    """Evaluate validated ordinary-float coefficients on an angular FFT grid."""
    if (
        isinstance(side, (bool, np.bool_))
        or not isinstance(side, (int, np.integer))
        or side < 1
        or not np.isfinite(pixel_mas)
        or pixel_mas <= 0
    ):
        raise ValueError("FFT side must be a positive integer and pixel scale positive")
    covariance = np.asarray(covariance, dtype=float)
    if covariance.shape != (2, 2) or not np.isfinite(covariance).all():
        raise ValueError("Blur covariance must be finite and two-by-two")
    covariance = (covariance + covariance.T) / 2
    if np.linalg.eigvalsh(covariance).min() < 0:
        raise ValueError("Blur covariance must be positive semidefinite")
    x, y = np.meshgrid(np.fft.fftfreq(side, pixel_mas), np.fft.fftfreq(side, pixel_mas))
    exponent = (
        -2
        * np.pi**2
        * (
            covariance[0, 0] * x**2
            + 2 * covariance[0, 1] * x * y
            + covariance[1, 1] * y**2
        )
    )
    weight = parameters["mixture_weight"]
    narrow = parameters[
        "mixture_single_scale" if ngs_count == 1 else "mixture_multi_scale"
    ]
    broad = (1 - weight * narrow) / (1 - weight)
    return weight * np.exp(narrow * exponent) + (1 - weight) * np.exp(broad * exponent)
