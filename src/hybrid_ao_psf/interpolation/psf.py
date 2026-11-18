"""Interpolate PSF images over a rectangular field grid.

Image interpolation uses the field and any varying airmass and wavelength axes.
Pixel scale varies only with airmass and wavelength. Saved files carry
the fitted image grid and its metadata for reuse.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import copy
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np
from ao_stats import (
    EE_GEOMETRY_ENCIRCLED,
    PsfMetadata,
    compute_psf_ee,
    compute_psf_fwhm,
)
from astropy import units as u
from numpy.typing import DTypeLike
from scipy.interpolate import RegularGridInterpolator

from .._units import _PhysicalRecord, quantity_value, require_quantity
from .._validation import (
    coarsest_precision,
    coordinate_precision,
    validate_psf_dtype,
)
from ..contracts import PsfPrediction
from ._core import (
    axis_index,
    cast_interpolator_value,
    clip_rectangular_field_coordinates,
    field_coordinates,
    file_grid_precision,
    grid_field_values,
    interpolation_axis_weights,
    interpolator_views,
    load_file_contents,
    make_read_only,
    rectangular_field_axes,
    require_finite_vector,
    require_positive_scalar,
    require_positive_vector,
    require_pupil,
    save_file_contents,
    unique_sorted,
    validate_file_format,
    validate_rectangular_field_coordinates,
    zenith_angle_to_airmass,
)

# Retain the file type identifier used by existing PSF files.
PSF_FILE_KIND = "hybrid_ao_psf_science_ho_psf_interpolator"
PSF_FILE_VERSION = 3
PSF_FIELD_COORDINATE_ATOL = 1.0e-6 * u.arcsec
COORD_AIRMASS = "airmass"
COORD_WAVELENGTH = "wavelength"
COORD_Y = "y"
COORD_X = "x"
FIELD_COORDINATE_ORDER = (COORD_Y, COORD_X)
VALID_COORDINATE_ORDERS = frozenset(
    {
        FIELD_COORDINATE_ORDER,
        (COORD_WAVELENGTH, *FIELD_COORDINATE_ORDER),
        (COORD_AIRMASS, *FIELD_COORDINATE_ORDER),
        (
            COORD_AIRMASS,
            COORD_WAVELENGTH,
            *FIELD_COORDINATE_ORDER,
        ),
    }
)
_AXIS_ATOL = 1.0e-10


@dataclass(frozen=True, kw_only=True)
class PsfSamples:
    """PSF samples used to build an interpolation file.

    Each PSF cube contains the images at all field points for one wavelength
    and zenith angle. Image interpolation accepts a shared wavelength and
    pixel scale or one value of each per cube. Metric measurement also
    accepts one value per PSF.

    Attributes:
        x: Field x-coordinates in arcseconds.
        y: Field y-coordinates in arcseconds.
        psfs: One PSF cube with shape ``(field_points, rows, columns)`` or
            a stack with shape ``(cubes, field_points, rows, columns)``. Use
            the same field points in the same order in every cube. Image
            columns increase toward ``+x`` and rows toward ``+y``.
        wavelength: PSF wavelength compatible with micrometres.
        pixel_scale: Angular pixel scale compatible with milliarcseconds.
        tel_diameter: Telescope diameter in metres.
        tel_pupil: Shared, dimensionless two-dimensional telescope pupil.
        zenith_angle: Optional scalar fixed zenith angle or one value per
            cube. Multiple unique values activate airmass interpolation.
    """

    x: u.Quantity
    y: u.Quantity
    psfs: np.ndarray
    wavelength: u.Quantity
    pixel_scale: u.Quantity
    tel_diameter: u.Quantity
    tel_pupil: u.Quantity
    zenith_angle: u.Quantity | None = None


@dataclass(frozen=True)
class PsfComparisonSummary:
    """Residuals from comparing predicted images with supplied reference samples.

    Attributes:
        psf_nrms_mean: Mean over images of the pixel residual RMS divided by
            the corresponding source image's pixel RMS.
        psf_nrms_max: Maximum of those normalized image residuals.
        pixel_scale_max_abs_error: Largest absolute pixel scale error in milliarcseconds.
        metric_rms_error: RMS residual by metric. FWHM is in milliarcseconds and
            encircled energy is dimensionless.
        metric_max_abs_error: Maximum absolute residual in the same metric units.
        num_cubes: Number of PSF cubes compared.
        num_points: Number of field points per cube.
    """

    psf_nrms_mean: float
    psf_nrms_max: float
    pixel_scale_max_abs_error: u.Quantity
    metric_rms_error: dict[str, u.Quantity]
    metric_max_abs_error: dict[str, u.Quantity]
    num_cubes: int
    num_points: int


@dataclass(frozen=True)
class PsfInterpolator(_PhysicalRecord):
    """Read-only rectangular PSF grid built from samples or loaded from disk.

    Physical coordinates with one sampled value stay fixed. Coordinates that
    vary add array axes before the field axes. Loaded interpolators are ready
    to evaluate requested coordinates.

    Attributes:
        coordinate_order: Array axes of ``psf_grid``, in order. The optional
            airmass and wavelength axes precede the field axes ``y`` and ``x``.
        zenith_angle_axis: Sampled zenith angles in degrees. Empty when the
            source provides no zenith-angle samples.
        airmass_axis: Dimensionless airmasses corresponding to the zenith angles.
        wavelength_axis: Sampled wavelengths in micrometres.
        x: Rectangular field x-axis in arcseconds.
        y: Rectangular field y-axis in arcseconds.
        psf_shape: Number of rows and columns in each PSF image.
        pixel_scale_grid: Angular pixel scale in milliarcseconds on the varying
            physical axes. A scalar when both physical coordinates are fixed.
        tel_diameter: Telescope diameter in metres.
        tel_pupil: Shared, dimensionless two-dimensional telescope pupil.
        psf_grid: PSF pixels indexed by ``coordinate_order`` and then by image
            row and column.
        grid_precision: Coordinate precision used for rounding at grid boundaries.
        dtype: Float32 by default or explicit float64 for pixels and predictions.
    """

    coordinate_order: tuple[str, ...]
    zenith_angle_axis: u.Quantity
    airmass_axis: u.Quantity
    wavelength_axis: u.Quantity
    x: u.Quantity
    y: u.Quantity
    psf_shape: tuple[int, int]
    pixel_scale_grid: u.Quantity
    tel_diameter: u.Quantity
    tel_pupil: u.Quantity
    psf_grid: np.ndarray
    grid_precision: DTypeLike | None = field(default=None, kw_only=True)
    dtype: DTypeLike = field(default=np.float32, kw_only=True)
    _cached_evaluator: RegularGridInterpolator | None = field(
        default=None, init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "coordinate_order", tuple(self.coordinate_order))
        object.__setattr__(
            self, "psf_shape", tuple(int(value) for value in self.psf_shape)
        )
        dtype = validate_psf_dtype(self.dtype)
        object.__setattr__(self, "dtype", dtype)
        object.__setattr__(
            self,
            "psf_grid",
            make_read_only(np.asarray(self.psf_grid, dtype=dtype)),
        )
        object.__setattr__(
            self, "grid_precision", coarsest_precision(self.dtype, self.grid_precision)
        )
        for name in (
            "zenith_angle_axis",
            "airmass_axis",
            "wavelength_axis",
            "x",
            "y",
            "pixel_scale_grid",
            "tel_diameter",
            "tel_pupil",
        ):
            object.__setattr__(self, name, make_read_only(getattr(self, name)))

    def __getattribute__(self, name: str) -> Any:
        value = super().__getattribute__(name)
        return interpolator_views(value) if name == "psf_grid" else value

    def _with_dtype(self, dtype: DTypeLike) -> PsfInterpolator:
        """Copy fitted values at dtype, retaining coordinate rounding precision."""
        dtype = validate_psf_dtype(dtype)
        names = (
            "zenith_angle_axis",
            "airmass_axis",
            "wavelength_axis",
            "x",
            "y",
            "pixel_scale_grid",
            "tel_diameter",
            "tel_pupil",
            "psf_grid",
        )
        return replace(
            self,
            dtype=dtype,
            grid_precision=coarsest_precision(self.grid_precision, dtype),
            **{
                name: cast_interpolator_value(getattr(self, name), dtype)
                for name in names
            },
        )

    def _get_evaluator(self) -> RegularGridInterpolator:
        """Reuse the saved evaluator or build and cache it on first use."""
        evaluator = self._cached_evaluator
        if evaluator is None:
            evaluator = _build_evaluator(self)
            object.__setattr__(self, "_cached_evaluator", evaluator)
        return evaluator

    def evaluate(
        self,
        *,
        zenith_angle: u.Quantity | None = None,
        wavelength: u.Quantity | None = None,
        x: u.Quantity,
        y: u.Quantity,
    ) -> PsfPrediction:
        """Evaluate one PSF image per paired field coordinate.

        Args:
            zenith_angle: Common scalar angle. Required for a varying airmass
                axis; may be omitted for a fixed or unspecified zenith angle.
            wavelength: Common scalar wavelength. Required for a varying
                wavelength axis; may be omitted for a fixed wavelength.
            x: Field x-coordinates as an angular quantity.
            y: Field y-coordinates as an angular quantity, paired with ``x``.

        Returns:
            PSF images in requested point order with a shared pixel scale and metadata.

        Raises:
            TypeError: If a required coordinate lacks an Astropy quantity.
            ValueError: If the units, shapes or requested coordinates are invalid.
        """
        return _evaluate_psf_interpolator(
            self,
            zenith_angle=zenith_angle,
            wavelength=wavelength,
            x=x,
            y=y,
        )


# Image and metadata validation


def _validate_psf_flux(psfs: np.ndarray, *, label: str) -> None:
    flux = np.sum(np.asarray(psfs, dtype=np.float64), axis=(-2, -1))
    if not np.all(np.isfinite(flux)):
        raise ValueError(f"{label} must have finite per-PSF total flux.")
    if np.any(flux <= 0.0):
        raise ValueError(f"{label} must have strictly positive per-PSF total flux.")


def _validate_source_psfs(
    psfs: Any, *, label: str, num_points: int, dtype: np.dtype
) -> np.ndarray:
    array = np.asarray(psfs, dtype=dtype)
    if array.ndim == 3:
        array = array[np.newaxis, ...]
    if array.ndim != 4:
        raise ValueError(
            f"{label} must have ndim=3 or ndim=4, got shape {array.shape}."
        )
    if array.shape[1] != int(num_points):
        raise ValueError(
            f"{label} point dimension must match {int(num_points)} field coordinates; got shape {array.shape}."
        )
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{label} must contain only finite values.")
    _validate_psf_flux(array, label=label)
    return array


def _validate_and_convert_cube_values(
    value: Any,
    *,
    unit: u.UnitBase,
    label: str,
    length: int,
    positive: bool,
    dtype=np.float64,
) -> u.Quantity:
    array = quantity_value(value, unit, label=label, dtype=dtype)
    if array.ndim == 0:
        vector = np.full(int(length), float(array), dtype=dtype)
    else:
        vector = array.reshape(-1)
        if vector.size != int(length):
            raise ValueError(
                f"{label} must be scalar or have one value per cube; got length {vector.size}."
            )
    if positive:
        vector = require_positive_vector(
            vector, label=label, length=length, dtype=dtype
        )
    else:
        vector = require_finite_vector(vector, label=label, length=length, dtype=dtype)
    return vector * unit


def _validate_and_convert_zenith_angles(
    value: Any, *, label: str, length: int, dtype=np.float64
) -> u.Quantity | None:
    if value is None:
        return None
    return _validate_and_convert_cube_values(
        value, unit=u.deg, label=label, length=length, positive=False
    )


def _validate_and_convert_psf_samples(
    samples: PsfSamples, *, dtype: np.dtype
) -> PsfSamples:
    """Validate image samples and express their physical metadata per cube."""
    if not isinstance(samples, PsfSamples):
        raise TypeError("samples must be a PsfSamples instance.")
    coordinates = field_coordinates(samples.x, samples.y)
    psfs = _validate_source_psfs(
        samples.psfs, label="psfs", num_points=coordinates.shape[0], dtype=dtype
    )
    num_cubes = psfs.shape[0]
    wavelength = _validate_and_convert_cube_values(
        samples.wavelength,
        unit=u.um,
        label="wavelength",
        length=num_cubes,
        positive=True,
        dtype=dtype,
    )
    pixel_scale = _validate_and_convert_cube_values(
        samples.pixel_scale,
        unit=u.mas,
        label="pixel_scale",
        length=num_cubes,
        positive=True,
        dtype=dtype,
    )
    zenith_angle = _validate_and_convert_zenith_angles(
        samples.zenith_angle,
        label="zenith_angle",
        length=num_cubes,
        dtype=dtype,
    )
    return PsfSamples(
        x=quantity_value(samples.x, u.arcsec, label="x", dtype=dtype).reshape(-1)
        * u.arcsec,
        y=quantity_value(samples.y, u.arcsec, label="y", dtype=dtype).reshape(-1)
        * u.arcsec,
        psfs=psfs,
        wavelength=wavelength,
        pixel_scale=pixel_scale,
        tel_diameter=require_positive_scalar(
            quantity_value(samples.tel_diameter, u.m, label="tel_diameter"),
            label="tel_diameter",
        )
        * u.m,
        tel_pupil=require_pupil(
            quantity_value(
                samples.tel_pupil, u.dimensionless_unscaled, label="tel_pupil"
            ),
            dtype=dtype,
        )
        * u.dimensionless_unscaled,
        zenith_angle=zenith_angle,
    )


# Grid axes and source cubes


def _validate_coordinate_order(coordinate_order: tuple[str, ...]) -> None:
    if tuple(coordinate_order) not in VALID_COORDINATE_ORDERS:
        raise ValueError(
            f"Unsupported PSF coordinate_order: {tuple(coordinate_order)!r}."
        )


def _coordinate_order_for_axes(
    zenith_axis: np.ndarray, wavelength_axis: np.ndarray
) -> tuple[str, ...]:
    order: list[str] = []
    if zenith_axis.size > 1:
        order.append(COORD_AIRMASS)
    if wavelength_axis.size > 1:
        order.append(COORD_WAVELENGTH)
    order.extend(FIELD_COORDINATE_ORDER)
    coordinate_order = tuple(order)
    _validate_coordinate_order(coordinate_order)
    return coordinate_order


def _varying_physical_shape(
    coordinate_order: tuple[str, ...],
    zenith_axis: np.ndarray,
    wavelength_axis: np.ndarray,
) -> tuple[int, ...]:
    """Return the shape of the varying airmass and wavelength axes."""
    shape: list[int] = []
    if COORD_AIRMASS in coordinate_order:
        shape.append(int(zenith_axis.size))
    if COORD_WAVELENGTH in coordinate_order:
        shape.append(int(wavelength_axis.size))
    return tuple(shape)


def _physical_grid_index(
    coordinate_order: tuple[str, ...],
    zenith_axis: np.ndarray,
    wavelength_axis: np.ndarray,
    zenith_angle: float | None,
    wavelength: float,
) -> tuple[int, ...]:
    index: list[int] = []
    if COORD_AIRMASS in coordinate_order:
        if zenith_angle is None:
            raise ValueError(
                "zenith_angle is required for varying airmass interpolation."
            )
        index.append(axis_index(zenith_axis, float(zenith_angle), label="zenith_angle"))
    if COORD_WAVELENGTH in coordinate_order:
        index.append(axis_index(wavelength_axis, float(wavelength), label="wavelength"))
    return tuple(index)


def _physical_grid_shape(interpolator: PsfInterpolator) -> tuple[int, ...]:
    return _varying_physical_shape(
        interpolator.coordinate_order,
        interpolator.zenith_angle_axis.to_value(u.deg),
        interpolator.wavelength_axis.to_value(u.um),
    )


def _interpolation_axes(
    interpolator: PsfInterpolator,
) -> tuple[np.ndarray, ...]:
    axes: list[np.ndarray] = []
    for name in interpolator.coordinate_order:
        if name == COORD_AIRMASS:
            axes.append(interpolator.airmass_axis.to_value(u.dimensionless_unscaled))
        elif name == COORD_WAVELENGTH:
            axes.append(interpolator.wavelength_axis.to_value(u.um))
        elif name == COORD_Y:
            axes.append(interpolator.y.to_value(u.arcsec))
        elif name == COORD_X:
            axes.append(interpolator.x.to_value(u.arcsec))
        else:
            raise ValueError(f"Unsupported PSF coordinate {name!r}.")
    return tuple(axes)


def _expected_psf_grid_shape(
    interpolator: PsfInterpolator,
) -> tuple[int, ...]:
    sizes = {
        COORD_AIRMASS: interpolator.airmass_axis.size,
        COORD_WAVELENGTH: interpolator.wavelength_axis.size,
        COORD_Y: interpolator.y.size,
        COORD_X: interpolator.x.size,
    }
    return tuple(int(sizes[name]) for name in interpolator.coordinate_order) + tuple(
        interpolator.psf_shape
    )


def _grid_psf_cubes(
    samples: PsfSamples,
    coordinate_order: tuple[str, ...],
    zenith_axis: np.ndarray,
    wavelength_axis: np.ndarray,
    field_x_axis: np.ndarray,
    field_y_axis: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Grid the images and record which cube supplies each airmass and wavelength pair."""
    dtype = samples.psfs.dtype
    physical_shape = _varying_physical_shape(
        coordinate_order, zenith_axis, wavelength_axis
    )
    source_cube_indices = np.full(physical_shape, -1, dtype=int)
    psf_grid = np.full(
        (
            *physical_shape,
            field_y_axis.size,
            field_x_axis.size,
            *samples.psfs.shape[2:],
        ),
        np.nan,
        dtype=dtype,
    )

    for cube_index in range(samples.psfs.shape[0]):
        grid_index = _physical_grid_index(
            coordinate_order,
            zenith_axis,
            wavelength_axis,
            None
            if samples.zenith_angle is None
            else float(samples.zenith_angle[cube_index].to_value(u.deg)),
            float(samples.wavelength[cube_index].to_value(u.um)),
        )
        if source_cube_indices[grid_index] != -1:
            raise ValueError("Duplicate PSF cube for varying physical coordinates.")
        source_cube_indices[grid_index] = cube_index
        psf_grid[grid_index] = grid_field_values(
            samples.x,
            samples.y,
            samples.psfs[cube_index],
            field_x_axis,
            field_y_axis,
            label="PSF sample",
            dtype=dtype,
        )

    if np.any(source_cube_indices < 0):
        raise ValueError(
            "PSF samples must form a complete varying physical-coordinate grid."
        )
    return psf_grid, source_cube_indices


# Requested coordinates and pixel scale


def _validate_fixed_value(
    value: float, fixed: float, *, label: str, dtype=np.float64
) -> None:
    if not np.isclose(
        np.dtype(dtype).type(value),
        np.dtype(dtype).type(fixed),
        rtol=2 * np.finfo(dtype).eps if np.dtype(dtype).itemsize == 4 else 0.0,
        atol=_AXIS_ATOL,
    ):
        raise ValueError(
            f"{label}={value} does not match this interpolator's fixed {label}={fixed}."
        )


def _validate_field_coordinates(
    interpolator: PsfInterpolator, coordinates: np.ndarray
) -> None:
    validate_rectangular_field_coordinates(
        interpolator.x.to_value(u.arcsec),
        interpolator.y.to_value(u.arcsec),
        coordinates,
        label="PSF",
        atol=PSF_FIELD_COORDINATE_ATOL.to_value(u.arcsec),
    )


def _clip_field_coordinates(
    interpolator: PsfInterpolator, coordinates: np.ndarray
) -> np.ndarray:
    """Clip validated coordinates to the interpolator's field bounds."""
    return clip_rectangular_field_coordinates(
        interpolator.x.to_value(u.arcsec),
        interpolator.y.to_value(u.arcsec),
        coordinates,
    )


def _physical_coordinate_values(
    interpolator: PsfInterpolator,
    *,
    zenith_angle: u.Quantity | None,
    wavelength: u.Quantity | None,
) -> dict[str, float]:
    """Resolve physical coordinates and snap rounding-level endpoint offsets."""
    values: dict[str, float] = {}
    if COORD_AIRMASS in interpolator.coordinate_order:
        if zenith_angle is None:
            raise ValueError("zenith_angle is required by this PSF file.")
        values[COORD_AIRMASS] = float(
            zenith_angle_to_airmass(
                require_quantity(zenith_angle, u.deg, label="zenith_angle")
            ).value
        )
    elif zenith_angle is not None and interpolator.zenith_angle_axis.size == 1:
        _validate_fixed_value(
            float(
                require_quantity(zenith_angle, u.deg, label="zenith_angle").to_value(
                    u.deg
                )
            ),
            float(interpolator.zenith_angle_axis[0].to_value(u.deg)),
            label="zenith_angle",
            dtype=interpolator.grid_precision,
        )

    if COORD_WAVELENGTH in interpolator.coordinate_order:
        if wavelength is None:
            raise ValueError("wavelength is required by this PSF file.")
        values[COORD_WAVELENGTH] = require_positive_scalar(
            quantity_value(wavelength, u.um, label="wavelength"), label="wavelength"
        )
    else:
        fixed_wavelength = float(interpolator.wavelength_axis[0].to_value(u.um))
        if wavelength is not None:
            _validate_fixed_value(
                require_positive_scalar(
                    quantity_value(wavelength, u.um, label="wavelength"),
                    label="wavelength",
                ),
                fixed_wavelength,
                label="wavelength",
                dtype=interpolator.grid_precision,
            )
        values[COORD_WAVELENGTH] = fixed_wavelength
    # Coordinate bounds include rounding at both source and evaluation precision.
    precision = interpolator.grid_precision
    for name, axis in (
        (COORD_AIRMASS, interpolator.airmass_axis),
        (COORD_WAVELENGTH, interpolator.wavelength_axis.to_value(u.um)),
    ):
        if name not in values or name not in interpolator.coordinate_order:
            continue
        bound = (
            2
            * np.finfo(precision).eps
            * max(1.0, abs(float(axis[0])), abs(float(axis[-1])))
        )
        if values[name] < axis[0] - bound or values[name] > axis[-1] + bound:
            raise ValueError(f"{name} outside calibrated range")
        values[name] = interpolator.dtype.type(np.clip(values[name], axis[0], axis[-1]))
    return values


def _evaluation_points(
    interpolator: PsfInterpolator,
    physical_values: Mapping[str, float],
    coordinates: np.ndarray,
) -> np.ndarray:
    columns: list[np.ndarray] = []
    for name in interpolator.coordinate_order:
        if name == COORD_AIRMASS:
            columns.append(
                np.full(
                    coordinates.shape[0],
                    float(physical_values[COORD_AIRMASS]),
                    dtype=float,
                )
            )
        elif name == COORD_WAVELENGTH:
            columns.append(
                np.full(
                    coordinates.shape[0],
                    float(physical_values[COORD_WAVELENGTH]),
                    dtype=float,
                )
            )
        elif name == COORD_Y:
            columns.append(coordinates[:, 1])
        elif name == COORD_X:
            columns.append(coordinates[:, 0])
        else:
            raise ValueError(f"Unsupported PSF coordinate {name!r}.")
    return np.column_stack(columns).astype(interpolator.dtype)


def _extend_weights(
    current: list[tuple[float, tuple[int, ...]]],
    axis_weights: list[tuple[int, float]],
) -> list[tuple[float, tuple[int, ...]]]:
    return [
        (float(weight * axis_weight), (*index, int(axis_index_value)))
        for weight, index in current
        for axis_index_value, axis_weight in axis_weights
    ]


def _cube_weights(
    interpolator: PsfInterpolator,
    physical_values: Mapping[str, float],
) -> list[tuple[float, tuple[int, ...]]]:
    """Pair products of linear axis weights with their physical grid indices."""
    weights: list[tuple[float, tuple[int, ...]]] = [(1.0, ())]
    if COORD_AIRMASS in interpolator.coordinate_order:
        weights = _extend_weights(
            weights,
            interpolation_axis_weights(
                interpolator.airmass_axis,
                physical_values[COORD_AIRMASS],
                label="airmass",
            ),
        )
    if COORD_WAVELENGTH in interpolator.coordinate_order:
        weights = _extend_weights(
            weights,
            interpolation_axis_weights(
                interpolator.wavelength_axis,
                physical_values[COORD_WAVELENGTH],
                label="wavelength",
            ),
        )
    return weights


def _interpolate_pixel_scale(
    interpolator: PsfInterpolator, physical_values: Mapping[str, float]
) -> u.Quantity:
    """Interpolate the angular pixel scale over the airmass and wavelength axes."""
    weights = _cube_weights(interpolator, physical_values)
    pixel_scales_mas = interpolator.pixel_scale_grid.to_value(u.mas)
    pixel_scale_mas = 0.0
    for weight, grid_index in weights:
        pixel_scale_mas += float(weight) * float(pixel_scales_mas[grid_index])
    return pixel_scale_mas * u.mas


# Evaluator construction and evaluation


def _build_evaluator(interpolator: PsfInterpolator) -> RegularGridInterpolator:
    """Build bounded linear evaluation over the validated, read-only pixel grid."""
    _validate_interpolator(interpolator)
    return RegularGridInterpolator(
        _interpolation_axes(interpolator),
        np.asarray(interpolator.psf_grid),
        method="linear",
        bounds_error=True,
    )


def _evaluate_psf_interpolator(
    interpolator: PsfInterpolator,
    *,
    zenith_angle: u.Quantity | None = None,
    wavelength: u.Quantity | None = None,
    x: u.Quantity,
    y: u.Quantity,
) -> PsfPrediction:
    """Evaluate images and physical metadata using the interpolator's cached evaluator."""
    evaluator = interpolator._get_evaluator()
    coordinates = field_coordinates(x, y)
    _validate_field_coordinates(interpolator, coordinates)
    coordinates = _clip_field_coordinates(interpolator, coordinates)
    physical_values = _physical_coordinate_values(
        interpolator,
        zenith_angle=zenith_angle,
        wavelength=wavelength,
    )
    points = _evaluation_points(interpolator, physical_values, coordinates)
    psfs = np.asarray(evaluator(points), dtype=interpolator.dtype).reshape(
        coordinates.shape[0],
        *interpolator.psf_shape,
    )
    pixel_scale = _interpolate_pixel_scale(interpolator, physical_values)
    _validate_psf_flux(psfs, label="predicted psfs")
    wavelength = float(physical_values[COORD_WAVELENGTH]) * u.um
    return PsfPrediction(
        psfs=psfs,
        dtype=interpolator.dtype,
        pixel_scale=pixel_scale,
        metadata=PsfMetadata(
            wavelength=wavelength,
            pixel_scale=pixel_scale,
            tel_diameter=interpolator.tel_diameter,
            tel_pupil=interpolator.tel_pupil,
        ),
    )


# Comparison residuals


def _psf_nrms(reference: np.ndarray, measured: np.ndarray) -> np.ndarray:
    numerator = np.sqrt(
        np.mean((np.asarray(measured) - np.asarray(reference)) ** 2, axis=(-2, -1))
    )
    denominator = np.sqrt(np.mean(np.asarray(reference) ** 2, axis=(-2, -1)))
    return numerator / denominator


def _metric_errors(
    reference_psfs: np.ndarray,
    measured_psfs: np.ndarray,
    *,
    wavelength: u.Quantity,
    reference_pixel_scale: float,
    prediction_metadata: PsfMetadata,
    tel_diameter: float,
    tel_pupil: u.Quantity,
) -> dict[str, np.ndarray]:
    dtype = reference_psfs.dtype
    reference_metadata = PsfMetadata(
        wavelength=wavelength.to(u.um),
        pixel_scale=float(reference_pixel_scale) * u.mas,
        tel_diameter=float(tel_diameter) * u.m,
        tel_pupil=tel_pupil.to(u.dimensionless_unscaled),
    )
    reference_fwhm = np.asarray(
        compute_psf_fwhm(
            reference_psfs, reference_metadata, preprocess="default", dtype=dtype
        ),
        dtype=dtype,
    ).reshape(-1)
    measured_fwhm = np.asarray(
        compute_psf_fwhm(
            measured_psfs, prediction_metadata, preprocess="default", dtype=dtype
        ),
        dtype=dtype,
    ).reshape(-1)
    if not np.all(np.isfinite(reference_fwhm)) or not np.all(
        np.isfinite(measured_fwhm)
    ):
        raise ValueError("PSF comparison could not compute finite fwhm metrics.")

    ee_aperture_diameters = (
        np.asarray(2.0 * reference_fwhm, dtype=dtype).reshape(-1, 1) * u.mas
    )
    reference_ee = np.asarray(
        compute_psf_ee(
            reference_psfs,
            reference_metadata,
            ee_apertures=ee_aperture_diameters,
            ee_geometry=EE_GEOMETRY_ENCIRCLED,
            preprocess="default",
            dtype=dtype,
        ),
        dtype=dtype,
    ).reshape(-1)
    measured_ee = np.asarray(
        compute_psf_ee(
            measured_psfs,
            prediction_metadata,
            ee_apertures=ee_aperture_diameters,
            ee_geometry=EE_GEOMETRY_ENCIRCLED,
            preprocess="default",
            dtype=dtype,
        ),
        dtype=dtype,
    ).reshape(-1)
    if not np.all(np.isfinite(reference_ee)) or not np.all(np.isfinite(measured_ee)):
        raise ValueError("PSF comparison could not compute finite ee metrics.")
    return {
        "fwhm": measured_fwhm - reference_fwhm,
        "ee": measured_ee - reference_ee,
    }


# File contents and validation


def _validate_evaluator(
    interpolator: PsfInterpolator,
    evaluator: RegularGridInterpolator,
) -> None:
    """Check the saved SciPy axes and boundary policy against the file."""
    if evaluator.method != "linear" or evaluator.bounds_error is not True:
        raise ValueError("PSF evaluator must use bounded linear interpolation.")
    axes = _interpolation_axes(interpolator)
    if len(evaluator.grid) != len(axes) or any(
        not np.array_equal(stored, expected)
        for stored, expected in zip(evaluator.grid, axes, strict=True)
    ):
        raise ValueError("PSF evaluator axes do not match file metadata.")


def _validate_interpolator(interpolator: PsfInterpolator) -> None:
    """Check coordinates, physical metadata and the pixel grid as a single file."""
    if not isinstance(interpolator, PsfInterpolator):
        raise TypeError("interpolator must be a PsfInterpolator instance.")
    _validate_coordinate_order(tuple(interpolator.coordinate_order))
    if len(interpolator.psf_shape) != 2 or any(
        int(value) <= 0 for value in interpolator.psf_shape
    ):
        raise ValueError("metadata.psf_shape must contain two positive dimensions.")
    if interpolator.airmass_axis.shape != interpolator.zenith_angle_axis.shape:
        raise ValueError("airmass_axis shape must match zenith_angle_axis shape.")
    if COORD_AIRMASS in interpolator.coordinate_order:
        if interpolator.zenith_angle_axis.size <= 1:
            raise ValueError(
                "varying airmass coordinate requires multiple zenith_angle_axis values."
            )
    elif interpolator.zenith_angle_axis.size > 1:
        raise ValueError(
            "multiple zenith_angle_axis values require varying airmass coordinate."
        )
    if COORD_WAVELENGTH in interpolator.coordinate_order:
        if interpolator.wavelength_axis.size <= 1:
            raise ValueError(
                "varying wavelength coordinate requires multiple wavelength_axis values."
            )
    elif interpolator.wavelength_axis.size != 1:
        raise ValueError(
            "fixed wavelength files must store exactly one wavelength_axis value."
        )
    if interpolator.x.ndim != 1 or interpolator.y.ndim != 1:
        raise ValueError("x and y must be 1-D rectangular field axes.")
    if interpolator.x.size == 0 or interpolator.y.size == 0:
        raise ValueError("x and y field axes must not be empty.")
    if np.any(np.diff(interpolator.x) <= 0.0) or np.any(np.diff(interpolator.y) <= 0.0):
        raise ValueError("x and y field axes must be strictly increasing.")
    physical_shape = _physical_grid_shape(interpolator)
    pixel_scale_grid = quantity_value(
        interpolator.pixel_scale_grid, u.mas, label="pixel_scale_grid", dtype=float
    )
    if pixel_scale_grid.shape != physical_shape:
        raise ValueError(f"pixel_scale_grid shape must be {physical_shape}.")
    if not np.all(np.isfinite(pixel_scale_grid)):
        raise ValueError("pixel_scale_grid must contain only finite values.")
    expected_psf_grid_shape = _expected_psf_grid_shape(interpolator)
    psf_grid = np.asarray(interpolator.psf_grid)
    if psf_grid.dtype != validate_psf_dtype(interpolator.dtype):
        raise ValueError("psf_grid precision must match dtype.")
    if psf_grid.shape != expected_psf_grid_shape:
        raise ValueError(
            f"psf_grid shape must be {expected_psf_grid_shape}; got {psf_grid.shape}."
        )
    if not np.all(np.isfinite(psf_grid)):
        raise ValueError("psf_grid must contain only finite values.")
    _validate_psf_flux(psf_grid, label="psf_grid")


def _interpolation_method(coordinate_order: tuple[str, ...]) -> str:
    label_by_coordinate = {
        COORD_AIRMASS: "airmass",
        COORD_WAVELENGTH: "wavelength",
        COORD_Y: "y",
        COORD_X: "x",
    }
    return "regular_grid_linear_" + "_".join(
        label_by_coordinate[name] for name in coordinate_order
    )


def _to_file_contents(interpolator: PsfInterpolator) -> dict[str, Any]:
    """Build file contents with one saved copy of the SciPy pixel grid."""
    if not isinstance(interpolator, PsfInterpolator):
        raise TypeError("interpolator must be a PsfInterpolator instance.")
    _validate_interpolator(interpolator)
    evaluator = copy(interpolator._get_evaluator())
    # SciPy has moved this helper between private modules; NumPy input is the
    # supported coordinate contract and keeps the pickle usable across that move.
    evaluator._asarray = np.asarray
    return {
        "kind": PSF_FILE_KIND,
        "version": PSF_FILE_VERSION,
        "grid_precision": interpolator.grid_precision.name,
        "interpolation": {
            "method": _interpolation_method(interpolator.coordinate_order),
            "coordinate_order": tuple(interpolator.coordinate_order),
        },
        "metadata": {
            "zenith_angle_axis": interpolator.zenith_angle_axis,
            "airmass_axis": interpolator.airmass_axis,
            "wavelength_axis": interpolator.wavelength_axis,
            "x": interpolator.x,
            "y": interpolator.y,
            "psf_shape": tuple(interpolator.psf_shape),
            "pixel_scale_grid": interpolator.pixel_scale_grid,
            "tel_diameter": interpolator.tel_diameter,
            "tel_pupil": interpolator.tel_pupil,
        },
        "model": {"prepared_psf_interpolator": evaluator},
    }


def _from_file_contents(file_contents: Mapping[str, Any]) -> PsfInterpolator:
    """Build an image interpolator from checked file contents."""
    validate_file_format(
        file_contents,
        kind=PSF_FILE_KIND,
        versions=(PSF_FILE_VERSION,),
    )
    interpolation = dict(file_contents.get("interpolation", {}))
    coordinate_order = tuple(
        str(value) for value in interpolation.get("coordinate_order", ())
    )
    _validate_coordinate_order(coordinate_order)
    if interpolation.get("method") != _interpolation_method(coordinate_order):
        raise ValueError("Unsupported PSF interpolation method.")
    interpolation_data = dict(file_contents.get("model", {}))
    evaluator = interpolation_data.get("prepared_psf_interpolator")
    if not isinstance(evaluator, RegularGridInterpolator):
        raise TypeError("PSF file requires a RegularGridInterpolator evaluator.")
    psf_grid = np.asarray(evaluator.values)
    dtype = validate_psf_dtype(psf_grid.dtype)
    metadata = dict(file_contents.get("metadata", {}))
    interpolator = PsfInterpolator(
        coordinate_order=coordinate_order,
        zenith_angle_axis=require_quantity(
            metadata.get("zenith_angle_axis"), u.deg, label="metadata.zenith_angle_axis"
        ),
        airmass_axis=require_quantity(
            metadata.get("airmass_axis"), u.one, label="metadata.airmass_axis"
        ),
        wavelength_axis=require_quantity(
            metadata.get("wavelength_axis"), u.um, label="metadata.wavelength_axis"
        ),
        x=require_quantity(metadata.get("x"), u.arcsec, label="metadata.x"),
        y=require_quantity(metadata.get("y"), u.arcsec, label="metadata.y"),
        psf_shape=tuple(int(value) for value in metadata.get("psf_shape", ())),
        pixel_scale_grid=require_quantity(
            metadata.get("pixel_scale_grid"), u.mas, label="metadata.pixel_scale_grid"
        ),
        tel_diameter=require_quantity(
            metadata.get("tel_diameter"), u.m, label="metadata.tel_diameter"
        ),
        tel_pupil=require_quantity(
            metadata.get("tel_pupil"), u.one, label="metadata.tel_pupil"
        ),
        psf_grid=psf_grid,
        dtype=dtype,
        grid_precision=file_grid_precision(file_contents),
    )
    _validate_interpolator(interpolator)
    _validate_evaluator(interpolator, evaluator)
    if not np.shares_memory(evaluator.values, interpolator.psf_grid):
        # A pickle may contain writable storage. Attach a fresh linear
        # interpolator to the validated read-only grid in that case.
        evaluator = _build_evaluator(interpolator)
    object.__setattr__(interpolator, "_cached_evaluator", evaluator)
    return interpolator


# Build an interpolator


def build_psf_interpolator(
    samples: PsfSamples,
    *,
    dtype: DTypeLike = np.float32,
) -> PsfInterpolator:
    """Build a linear interpolator from a complete rectangular field grid.

    An axis with multiple sampled values becomes an interpolation coordinate.
    Omitting zenith angles leaves field position and wavelength as the only
    coordinate constraints.

    Args:
        samples: One PSF cube for each sampled zenith angle and wavelength
            combination. The PSFs are used at their supplied
            flux and in their supplied orientation.
        dtype: Precision for fitted and returned pixels, float32 or float64.

    Returns:
        A read-only PSF provider ready to evaluate or save.

    Raises:
        TypeError: If the samples or physical metadata use unsupported types.
        ValueError: If the field or physical grid is incomplete or duplicated,
            or if any units, PSF values or precision are invalid.
    """

    dtype = validate_psf_dtype(dtype)
    source_samples = samples
    samples = _validate_and_convert_psf_samples(samples, dtype=dtype)
    field_x_axis, field_y_axis = rectangular_field_axes(
        samples.x,
        samples.y,
        label="PSF samples",
    )
    wavelength_axis = unique_sorted(
        samples.wavelength.to_value(u.um), label="wavelength"
    )
    zenith_axis = (
        unique_sorted(samples.zenith_angle.to_value(u.deg), label="zenith_angle")
        if samples.zenith_angle is not None
        else np.asarray([], dtype=dtype)
    )
    coordinate_order = _coordinate_order_for_axes(zenith_axis, wavelength_axis)
    psf_grid, source_cube_indices = _grid_psf_cubes(
        samples,
        coordinate_order,
        zenith_axis,
        wavelength_axis,
        field_x_axis,
        field_y_axis,
    )
    pixel_scale_grid = samples.pixel_scale[source_cube_indices]

    return PsfInterpolator(
        coordinate_order=coordinate_order,
        zenith_angle_axis=zenith_axis * u.deg,
        airmass_axis=zenith_angle_to_airmass(zenith_axis * u.deg),
        wavelength_axis=wavelength_axis * u.um,
        x=np.asarray(field_x_axis, dtype=dtype) * u.arcsec,
        y=np.asarray(field_y_axis, dtype=dtype) * u.arcsec,
        psf_shape=tuple(int(v) for v in samples.psfs.shape[2:]),
        pixel_scale_grid=pixel_scale_grid,
        tel_diameter=samples.tel_diameter,
        tel_pupil=samples.tel_pupil,
        psf_grid=psf_grid,
        dtype=dtype,
        grid_precision=coordinate_precision(
            dtype,
            source_samples.x,
            source_samples.y,
            source_samples.zenith_angle,
            source_samples.wavelength,
        ),
    )._with_dtype(dtype)


# Compare reference cubes


def compare_psf_interpolator(
    interpolator: PsfInterpolator,
    samples: PsfSamples,
) -> PsfComparisonSummary:
    """Evaluate reference coordinates and compare the predicted and supplied PSFs.

    For each pair, use the source image's FWHM as the encircled-energy
    aperture radius for both images.

    Args:
        interpolator: Built or loaded PSF interpolator.
        samples: Reference images and coordinates for comparison.

    Returns:
        Image residual statistics and pixel-scale, FWHM and encircled-energy
        errors across the compared samples.

    Raises:
        TypeError: If the samples or physical metadata use unsupported types.
        ValueError: If samples are invalid, requested coordinates lie outside the coordinate ranges,
            or comparison cannot yield finite metrics.
    """

    samples = _validate_and_convert_psf_samples(samples, dtype=interpolator.dtype)
    residuals: list[np.ndarray] = []
    pixel_scale_errors: list[float] = []
    metric_errors: dict[str, list[np.ndarray]] = {"fwhm": [], "ee": []}
    for index in range(samples.psfs.shape[0]):
        wavelength = samples.wavelength[index]
        prediction = interpolator.evaluate(
            zenith_angle=None
            if samples.zenith_angle is None
            else samples.zenith_angle[index],
            wavelength=wavelength,
            x=samples.x,
            y=samples.y,
        )
        reference = samples.psfs[index]
        residuals.append(_psf_nrms(reference, prediction.psfs))
        pixel_scale_errors.append(
            abs(
                float(samples.pixel_scale[index].to_value(u.mas))
                - prediction.pixel_scale.to_value(u.mas)
            )
        )
        cube_metric_errors = _metric_errors(
            reference,
            prediction.psfs,
            wavelength=wavelength,
            reference_pixel_scale=float(samples.pixel_scale[index].to_value(u.mas)),
            prediction_metadata=prediction.metadata,
            tel_diameter=float(samples.tel_diameter.to_value(u.m)),
            tel_pupil=samples.tel_pupil,
        )
        for name, values in cube_metric_errors.items():
            metric_errors[name].append(values)
    residual = np.concatenate(residuals)
    metric_units = {"fwhm": u.mas, "ee": u.one}
    metric_error_arrays = {
        name: np.concatenate(values) for name, values in metric_errors.items()
    }
    return PsfComparisonSummary(
        psf_nrms_mean=float(np.mean(residual)),
        psf_nrms_max=float(np.max(residual)),
        pixel_scale_max_abs_error=float(np.max(pixel_scale_errors)) * u.mas,
        metric_rms_error={
            name: float(np.sqrt(np.mean(values**2))) * metric_units[name]
            for name, values in metric_error_arrays.items()
        },
        metric_max_abs_error={
            name: float(np.max(np.abs(values))) * metric_units[name]
            for name, values in metric_error_arrays.items()
        },
        num_cubes=int(samples.psfs.shape[0]),
        num_points=int(samples.psfs.shape[1]),
    )


# Save and load files


def save_psf_interpolator(
    interpolator: PsfInterpolator,
    path: Path,
    *,
    overwrite: bool = False,
) -> None:
    """Write a versioned pickle with the fitted image grid and its metadata.

    Create missing parent directories and save the file atomically.

    Args:
        interpolator: Fitted image provider to save.
        path: Destination pickle path.
        overwrite: Whether an existing file may be replaced.

    Raises:
        FileExistsError: If the destination exists and overwrite is false.
        TypeError: If the file type or metadata types are invalid.
        ValueError: If the file fails validation.
    """
    save_file_contents(_to_file_contents(interpolator), Path(path), overwrite=overwrite)


def load_psf_interpolator(
    path: Path, *, dtype: DTypeLike | None = None
) -> PsfInterpolator:
    """Load a fitted image grid and its saved SciPy evaluator.

    Pickle reconstruction can execute Python code. Load only trusted files.

    Args:
        path: Path to a saved PSF file.
        dtype: Optional float32 or float64 evaluation precision. The default
            preserves stored precision. Conversion retains coordinate rounding
            precision and does not change the saved file.

    Returns:
        A read-only provider at the selected precision, ready to evaluate.

    Raises:
        FileNotFoundError: If the file path does not exist.
        TypeError: If the file contents or metadata have invalid types.
        ValueError: If the file type, format version, method or fitted grid is invalid.
    """
    interpolator = _from_file_contents(load_file_contents(Path(path)))
    if dtype is not None and validate_psf_dtype(dtype) != interpolator.dtype:
        return interpolator._with_dtype(dtype)
    return interpolator
