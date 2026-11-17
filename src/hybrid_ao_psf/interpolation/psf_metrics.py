"""Measure and interpolate FWHM, encircled energy and Strehl ratio.

Measurements can come from PSFs or already tabulated samples. Interpolation
fits those measurements over the field and, when sampled, over airmass.
Rectangular grids use regular-grid interpolation. Scattered field points use
a scaled radial basis function (RBF) fit. The sample coordinates select the method.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Literal

import numpy as np
from ao_stats import (
    DEFAULT_EE_CENTER,
    DEFAULT_FWHM_METHOD,
    DEFAULT_FWHM_REFINEMENT_FACTOR,
    EE_GEOMETRY_ENCIRCLED,
    PsfMetadata,
    compute_psf_ee,
    compute_psf_fwhm,
    compute_psf_sr,
)
from astropy import units as u
from numpy.typing import DTypeLike
from scipy.interpolate import RegularGridInterpolator
from scipy.spatial import Delaunay, QhullError

from .._units import _PhysicalRecord, quantity_value, require_quantity
from .._validation import (
    coarsest_precision,
    coordinate_precision,
    snap_coordinate_range,
    validate_psf_dtype,
)
from ..contracts import PsfMetricPrediction
from ._core import (
    axis_index,
    cast_interpolator_value,
    clip_rectangular_field_coordinates,
    evaluate_scaled_rbfs,
    field_coordinates,
    fit_scaled_rbfs,
    grid_field_values,
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
    validate_psf_array,
    validate_rectangular_field_coordinates,
    zenith_angle_to_airmass,
)
from .psf import PsfSamples

# Retain the file type identifier used by existing PSF metric files.
PSF_METRIC_FILE_KIND = "hybrid_ao_psf_ngs_ho_metric_interpolator"
PSF_METRIC_FILE_VERSION = 4
PSF_METRIC_INTERPOLATION_TYPE_REGULAR_GRID = "regular_grid"
PSF_METRIC_INTERPOLATION_TYPE_RBF = "rbf"
COORD_AIRMASS = "airmass"
COORD_X = "x"
COORD_Y = "y"
REGULAR_GRID_FIELD_ORDER = (COORD_Y, COORD_X)
RBF_FIELD_ORDER = (COORD_X, COORD_Y)
REGULAR_GRID_COORDINATE_ORDERS = frozenset(
    {
        REGULAR_GRID_FIELD_ORDER,
        (COORD_AIRMASS, *REGULAR_GRID_FIELD_ORDER),
    }
)
RBF_COORDINATE_ORDERS = frozenset(
    {
        RBF_FIELD_ORDER,
        (COORD_AIRMASS, *RBF_FIELD_ORDER),
    }
)
_METRIC_UNITS = {"ee": u.one, "fwhm": u.mas, "sr": u.one}
REQUIRED_PSF_METRICS = tuple(_METRIC_UNITS)
OPTIONAL_PSF_METRICS: tuple[str, ...] = ()
_FIELD_ATOL = 1.0e-10


@dataclass(frozen=True, kw_only=True)
class PsfMetricSamples:
    """Measured PSF metric samples for interpolation.

    Each metric accepts a vector of field points for one zenith angle or
    a two-dimensional array with one row per zenith angle.
    Multiple unique zenith angles activate the airmass axis. A single angle
    is stored as fixed-coordinate metadata.

    Attributes:
        x: Field x-coordinates compatible with arcseconds.
        y: Field y-coordinates compatible with arcseconds.
        ee: Dimensionless enclosed-energy quantities.
        fwhm: FWHM quantities compatible with milliarcseconds.
        sr: Dimensionless Strehl-ratio quantities.
        zenith_angle: Optional scalar angle or one angle per row,
            compatible with degrees.
    """

    x: u.Quantity
    y: u.Quantity
    ee: u.Quantity
    fwhm: u.Quantity
    sr: u.Quantity
    zenith_angle: u.Quantity | None = None


@dataclass(frozen=True)
class PsfMetricComparisonSummary:
    """Errors from comparing predicted metrics with supplied reference measurements.

    Encircled energy and Strehl residuals are dimensionless. FWHM residuals
    are in milliarcseconds.

    Attributes:
        metric_rms_error: Root-mean-square reference-sample residual by metric name.
        metric_max_abs_error: Maximum absolute reference-sample residual by metric name.
        num_planes: Number of compared sampled zenith-angle rows.
        num_points: Number of field points per sampled zenith angle.
    """

    metric_rms_error: dict[str, u.Quantity]
    metric_max_abs_error: dict[str, u.Quantity]
    num_planes: int
    num_points: int


@dataclass(frozen=True)
class PsfMetricInterpolator(_PhysicalRecord):
    """Read-only metric provider built from measurements or loaded from disk.

    Regular-grid providers store a grid per metric. RBF providers store the
    RBF evaluators and the coordinate scaling used to evaluate them.
    Interpolation data is private; callers build or load a provider and call
    evaluate().

    Attributes:
        coordinate_order: Interpolation-coordinate names.
        zenith_angle_axis: Supported zenith angles in degrees.
        airmass_axis: Dimensionless airmass values corresponding to the zenith
            axis.
        x: Rectangular x-axis or the original RBF sample x-coordinates in arcseconds.
        y: Rectangular y-axis or the original RBF sample y-coordinates in arcseconds.
        metric_names: Ordered names of the interpolated metrics.
        interpolation_type: Method selected from the sample coordinates: regular_grid or rbf.
        grid_precision: Coordinate precision used for rounding at grid boundaries.
        dtype: Float32 by default or explicit float64 for predictions.
    """

    coordinate_order: tuple[str, ...]
    zenith_angle_axis: u.Quantity
    airmass_axis: u.Quantity
    x: u.Quantity
    y: u.Quantity
    metric_names: tuple[str, ...]
    interpolation_type: str
    _interpolation_data: Mapping[str, Any] = field(repr=False)
    grid_precision: DTypeLike | None = field(default=None, kw_only=True)
    dtype: DTypeLike = field(default=np.float32, kw_only=True)

    def __post_init__(self) -> None:
        object.__setattr__(self, "dtype", validate_psf_dtype(self.dtype))
        object.__setattr__(self, "coordinate_order", tuple(self.coordinate_order))
        object.__setattr__(
            self, "metric_names", tuple(str(value) for value in self.metric_names)
        )
        object.__setattr__(
            self, "grid_precision", coarsest_precision(self.dtype, self.grid_precision)
        )
        for name in (
            "zenith_angle_axis",
            "airmass_axis",
            "x",
            "y",
            "_interpolation_data",
        ):
            object.__setattr__(self, name, make_read_only(getattr(self, name)))

    def __getattribute__(self, name: str) -> Any:
        value = super().__getattribute__(name)
        return interpolator_views(value) if name == "_interpolation_data" else value

    def _with_dtype(self, dtype: DTypeLike) -> PsfMetricInterpolator:
        """Copy fitted values at dtype, retaining coordinate rounding precision."""
        dtype = validate_psf_dtype(dtype)
        names = ("zenith_angle_axis", "airmass_axis", "x", "y", "_interpolation_data")
        return replace(
            self,
            dtype=dtype,
            grid_precision=coarsest_precision(self.grid_precision, dtype),
            **{
                name: cast_interpolator_value(getattr(self, name), dtype)
                for name in names
            },
        )

    def evaluate(
        self,
        *,
        zenith_angle: u.Quantity | None = None,
        x: u.Quantity,
        y: u.Quantity,
    ) -> PsfMetricPrediction:
        """Evaluate metrics at paired field coordinates within the fitted range.

        Args:
            zenith_angle: Scalar angle or one angle per field point. May be
                omitted for a fixed or unspecified zenith angle.
            x: Field x-coordinates as an angular quantity.
            y: Field y-coordinates as an angular quantity, paired with ``x``.

        Returns:
            Encircled energy and Strehl as dimensionless quantities, with
            FWHM in milliarcseconds, in requested point order.

        Raises:
            TypeError: If a required coordinate lacks an Astropy quantity.
            ValueError: If the units, shapes or requested coordinates are invalid,
                or the fitted values fail metric validation.
        """
        return _evaluate_psf_metric_interpolator(
            self,
            zenith_angle=zenith_angle,
            x=x,
            y=y,
        )


# Source image preparation


def _validate_and_convert_zenith_angles(
    value: Any, *, label: str, length: int, dtype=np.float64
) -> u.Quantity | None:
    if value is None:
        return None
    array = quantity_value(value, u.deg, label=label, dtype=dtype)
    if array.ndim == 0:
        vector = np.full(int(length), float(array), dtype=dtype)
    else:
        vector = array.reshape(-1)
        if vector.size != int(length):
            raise ValueError(
                f"{label} must be scalar or have {length} values; got {vector.size}."
            )
    return require_finite_vector(vector, label=label, length=length) * u.deg


def _expand_psf_metadata(
    value: Any, shape: tuple[int, int], *, unit: u.UnitBase, label: str
) -> float | np.ndarray:
    """Flatten metadata in cube order, followed by field-point order."""
    num_cubes, num_points = shape
    array = quantity_value(value, unit, label=label, dtype=float)
    if array.ndim == 0:
        scalar = require_positive_scalar(array, label=label)
        return scalar
    if array.shape == (num_cubes,):
        vector = require_positive_vector(array, label=label, length=num_cubes)
        return np.repeat(vector, num_points)
    if array.shape == shape:
        flat = require_positive_vector(
            array.reshape(-1), label=label, length=num_cubes * num_points
        )
        return flat
    flat = np.asarray(array, dtype=float).reshape(-1)
    if flat.size == num_cubes * num_points:
        return require_positive_vector(flat, label=label, length=num_cubes * num_points)
    raise ValueError(
        f"{label} must be scalar, per-cube, or per-PSF; got shape {array.shape}."
    )


def _measurement_metadata(samples: PsfSamples) -> PsfMetadata:
    """Expand cube metadata to match the flattened PSF array."""
    sample_shape = samples.psfs.shape[:2]
    return PsfMetadata(
        wavelength=_expand_psf_metadata(
            samples.wavelength, sample_shape, unit=u.um, label="wavelength"
        )
        * u.um,
        pixel_scale=_expand_psf_metadata(
            samples.pixel_scale, sample_shape, unit=u.mas, label="pixel_scale"
        )
        * u.mas,
        tel_diameter=samples.tel_diameter,
        tel_pupil=samples.tel_pupil,
    )


def _validate_and_convert_measurement_samples(
    samples: PsfSamples, *, dtype: DTypeLike = np.float32
) -> PsfSamples:
    if not isinstance(samples, PsfSamples):
        raise TypeError("samples must be a PsfSamples instance.")
    coordinates = field_coordinates(samples.x, samples.y)
    psfs = np.asarray(samples.psfs, dtype=dtype)
    if psfs.ndim == 3:
        psfs = psfs[np.newaxis, ...]
    psfs = validate_psf_array(psfs, label="psfs", ndim=4, dtype=dtype)
    if psfs.shape[1] != coordinates.shape[0]:
        raise ValueError(
            "psfs point dimension must match field coordinates; "
            f"got {psfs.shape} for {coordinates.shape[0]} points."
        )
    zenith_angle = _validate_and_convert_zenith_angles(
        samples.zenith_angle, label="zenith_angle", length=psfs.shape[0], dtype=dtype
    )
    _expand_psf_metadata(
        samples.wavelength, psfs.shape[:2], unit=u.um, label="wavelength"
    )
    _expand_psf_metadata(
        samples.pixel_scale, psfs.shape[:2], unit=u.mas, label="pixel_scale"
    )
    return PsfSamples(
        zenith_angle=zenith_angle,
        x=quantity_value(samples.x, u.arcsec, label="x", dtype=dtype).reshape(-1)
        * u.arcsec,
        y=quantity_value(samples.y, u.arcsec, label="y", dtype=dtype).reshape(-1)
        * u.arcsec,
        psfs=psfs,
        wavelength=require_quantity(samples.wavelength, u.um, label="wavelength"),
        pixel_scale=require_quantity(samples.pixel_scale, u.mas, label="pixel_scale"),
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
    )


# Metric sample preparation


def _infer_zenith_sample_count(samples: PsfMetricSamples, n_points: int) -> int:
    """Resolve zenith sample counts across metric arrays and zenith angles."""
    counts: list[int] = []
    for name in REQUIRED_PSF_METRICS:
        metric = quantity_value(
            getattr(samples, name), _METRIC_UNITS[name], label=name, dtype=float
        )
        if metric.ndim == 1:
            if metric.shape[0] != n_points:
                raise ValueError(
                    f"{name} must have length {n_points}; got shape {metric.shape}."
                )
            counts.append(1)
        elif metric.ndim == 2:
            if metric.shape[1] != n_points:
                raise ValueError(
                    f"{name} second dimension must be {n_points}; got shape {metric.shape}."
                )
            counts.append(int(metric.shape[0]))
        else:
            raise ValueError(
                f"{name} must have shape (points,) or (zenith samples, points); got {metric.shape}."
            )
    if samples.zenith_angle is not None:
        zenith = quantity_value(
            samples.zenith_angle, u.deg, label="zenith_angle", dtype=float
        )
        if zenith.ndim != 0:
            counts.append(int(zenith.reshape(-1).size))
    num_zenith_samples = max(counts) if counts else 1
    if any(count not in {1, num_zenith_samples} for count in counts):
        raise ValueError(
            "PSF metric zenith sample counts must match across metrics and zenith_angle."
        )
    return int(num_zenith_samples)


def _validate_and_convert_metric_array(
    value: Any,
    *,
    unit: u.UnitBase,
    label: str,
    expected_shape: tuple[int, int],
    dtype=np.float64,
) -> u.Quantity:
    if value is None:
        raise ValueError(f"{label} is required.")
    metric = quantity_value(value, unit, label=label, dtype=dtype)
    if metric.ndim == 1 and expected_shape[0] == 1:
        metric = metric[np.newaxis, :]
    if metric.shape != expected_shape:
        raise ValueError(
            f"{label} must have shape {expected_shape}, got {metric.shape}."
        )
    if not np.all(np.isfinite(metric)):
        raise ValueError(f"{label} must contain only finite values.")
    if np.any(metric <= 0.0):
        raise ValueError(f"{label} must contain only values > 0.")
    if label == "ee" and np.any(metric > 1.0):
        raise ValueError("ee must not contain values > 1.")
    return metric * unit


def _validate_and_convert_metric_samples(
    samples: PsfMetricSamples, *, dtype: DTypeLike = np.float32
) -> PsfMetricSamples:
    """Validate metric units and resolve a common zenith-by-field layout."""
    if not isinstance(samples, PsfMetricSamples):
        raise TypeError("samples must be a PsfMetricSamples instance.")
    coordinates = field_coordinates(samples.x, samples.y)
    num_zenith_samples = _infer_zenith_sample_count(samples, coordinates.shape[0])
    zenith_angle = _validate_and_convert_zenith_angles(
        samples.zenith_angle,
        label="zenith_angle",
        length=num_zenith_samples,
        dtype=dtype,
    )
    if zenith_angle is None and num_zenith_samples != 1:
        raise ValueError("Multiple PSF metric rows require zenith_angle.")
    expected_shape = (num_zenith_samples, coordinates.shape[0])
    metrics = {
        name: _validate_and_convert_metric_array(
            getattr(samples, name),
            unit=unit,
            label=name,
            expected_shape=expected_shape,
            dtype=dtype,
        )
        for name, unit in _METRIC_UNITS.items()
    }
    return PsfMetricSamples(
        zenith_angle=zenith_angle,
        x=quantity_value(samples.x, u.arcsec, label="x", dtype=dtype).reshape(-1)
        * u.arcsec,
        y=quantity_value(samples.y, u.arcsec, label="y", dtype=dtype).reshape(-1)
        * u.arcsec,
        **metrics,
    )


# Regular-grid and RBF fitting


def _zenith_axis(samples: PsfMetricSamples) -> np.ndarray:
    if samples.zenith_angle is None:
        return np.asarray([], dtype=float)
    return unique_sorted(samples.zenith_angle.to_value(u.deg), label="zenith_angle")


def _field_interpolation_type(samples: PsfMetricSamples) -> str:
    """Select a complete rectangular grid or scattered fit, rejecting duplicates first."""
    coordinates = field_coordinates(samples.x, samples.y)
    x_axis = unique_sorted(coordinates[:, 0], label="x")
    y_axis = unique_sorted(coordinates[:, 1], label="y")
    seen: set[tuple[int, int]] = set()
    for x, y in coordinates:
        point = (axis_index(x_axis, x, label="x"), axis_index(y_axis, y, label="y"))
        if point in seen:
            raise ValueError(
                f"Duplicate PSF metric samples field point at x={x}, y={y}."
            )
        seen.add(point)
    if len(seen) == x_axis.size * y_axis.size:
        return PSF_METRIC_INTERPOLATION_TYPE_REGULAR_GRID
    return PSF_METRIC_INTERPOLATION_TYPE_RBF


def _coordinate_order(
    interpolation_type: str, zenith_axis: np.ndarray
) -> tuple[str, ...]:
    field_order = (
        REGULAR_GRID_FIELD_ORDER
        if interpolation_type == PSF_METRIC_INTERPOLATION_TYPE_REGULAR_GRID
        else RBF_FIELD_ORDER
    )
    return (COORD_AIRMASS, *field_order) if zenith_axis.size > 1 else field_order


def _validate_coordinate_order_for_interpolation_type(
    coordinate_order: tuple[str, ...], interpolation_type: str
) -> None:
    if interpolation_type not in (
        PSF_METRIC_INTERPOLATION_TYPE_REGULAR_GRID,
        PSF_METRIC_INTERPOLATION_TYPE_RBF,
    ):
        raise ValueError(
            f"Unsupported PSF metric interpolation type: {interpolation_type!r}."
        )
    valid = (
        REGULAR_GRID_COORDINATE_ORDERS
        if interpolation_type == PSF_METRIC_INTERPOLATION_TYPE_REGULAR_GRID
        else RBF_COORDINATE_ORDERS
    )
    if tuple(coordinate_order) not in valid:
        raise ValueError(
            f"Unsupported PSF metric coordinate_order for {interpolation_type}: {tuple(coordinate_order)!r}."
        )
    if "wavelength" in coordinate_order:
        raise ValueError(
            "PSF metric files must not use wavelength as an interpolation coordinate."
        )


def _sample_coordinates(
    samples: PsfMetricSamples, coordinate_order: tuple[str, ...]
) -> np.ndarray:
    columns: list[np.ndarray] = []
    num_zenith_samples = samples.ee.shape[0]
    n_points = samples.x.size
    for name in coordinate_order:
        if name == COORD_AIRMASS:
            if samples.zenith_angle is None:
                raise ValueError(
                    "zenith_angle is required for varying airmass interpolation."
                )
            airmasses = zenith_angle_to_airmass(samples.zenith_angle).to_value(
                u.dimensionless_unscaled
            )
            columns.append(np.repeat(airmasses, n_points))
        elif name == COORD_X:
            columns.append(np.tile(samples.x.to_value(u.arcsec), num_zenith_samples))
        elif name == COORD_Y:
            columns.append(np.tile(samples.y.to_value(u.arcsec), num_zenith_samples))
        else:
            raise ValueError(f"Unsupported PSF metric coordinate {name!r}.")
    return np.column_stack(columns)


def _fit_rbf_metrics(
    samples: PsfMetricSamples,
    coordinate_order: tuple[str, ...],
) -> dict[str, Any]:
    """Fit each metric using the same scaled coordinates of the source samples."""
    coordinates = _sample_coordinates(samples, coordinate_order)
    metrics = {
        name: getattr(samples, name).to_value(unit).reshape(-1)
        for name, unit in _METRIC_UNITS.items()
    }
    return fit_scaled_rbfs(coordinates, metrics)


def _fit_regular_grid_metrics(
    samples: PsfMetricSamples,
    coordinate_order: tuple[str, ...],
    zenith_axis: np.ndarray,
) -> tuple[u.Quantity, u.Quantity, dict[str, Any]]:
    """Place metrics on rectangular field axes and order their zenith samples."""
    dtype = samples.ee.dtype
    _validate_coordinate_order_for_interpolation_type(
        coordinate_order, PSF_METRIC_INTERPOLATION_TYPE_REGULAR_GRID
    )
    if COORD_AIRMASS in coordinate_order:
        if zenith_axis.size != samples.ee.shape[0]:
            raise ValueError(
                "PSF metric samples contain duplicate zenith-angle samples."
            )
    elif samples.ee.shape[0] != 1:
        raise ValueError("Field-only PSF metric files require exactly one source row.")
    x_axis, y_axis = rectangular_field_axes(
        samples.x, samples.y, label="PSF metric samples"
    )
    metric_grids: dict[str, u.Quantity] = {}
    physical_shape = (zenith_axis.size,) if COORD_AIRMASS in coordinate_order else ()
    for name, unit in _METRIC_UNITS.items():
        values = np.asarray(getattr(samples, name).to_value(unit), dtype=dtype)
        metric_grid = np.full(
            (*physical_shape, y_axis.size, x_axis.size), np.nan, dtype=dtype
        )
        for zenith_index in range(values.shape[0]):
            grid_index = (
                (
                    axis_index(
                        zenith_axis,
                        float(samples.zenith_angle[zenith_index].to_value(u.deg)),
                        label="zenith_angle",
                    ),
                )
                if COORD_AIRMASS in coordinate_order
                else ()
            )
            metric_grid[grid_index] = grid_field_values(
                samples.x,
                samples.y,
                values[zenith_index],
                x_axis,
                y_axis,
                label=f"PSF metric {name}",
                dtype=dtype,
            )
        if not np.all(np.isfinite(metric_grid)):
            raise ValueError(f"PSF metric {name} grid is incomplete.")
        metric_grids[name] = metric_grid * unit
    return (
        np.asarray(x_axis, dtype=dtype) * u.arcsec,
        np.asarray(y_axis, dtype=dtype) * u.arcsec,
        {"metric_grids": metric_grids},
    )


# Requested coordinates and field region


def _validate_airmass_range(
    interpolator: PsfMetricInterpolator, airmass: np.ndarray, zenith_angle: np.ndarray
) -> np.ndarray:
    axis = interpolator.airmass_axis.to_value(u.dimensionless_unscaled).reshape(-1)
    airmass = np.asarray(airmass, dtype=axis.dtype)
    minimum = float(np.min(axis))
    maximum = float(np.max(axis))
    try:
        airmass[...] = snap_coordinate_range(
            airmass,
            [minimum, maximum],
            dtype=interpolator.grid_precision,
            label="zenith_angle",
        )
    except ValueError as error:
        supported = interpolator.zenith_angle_axis.to_value(u.deg)
        tolerance = (
            2
            * np.finfo(interpolator.grid_precision).eps
            * max(1.0, abs(minimum), abs(maximum))
        )
        outside = (airmass < minimum - tolerance) | (airmass > maximum + tolerance)
        requested = float(
            np.asarray(zenith_angle).reshape(-1)[np.flatnonzero(outside)[0]]
        )
        raise ValueError(
            f"zenith_angle={requested} deg is outside this interpolator's zenith_angle range [{supported[0]}, {supported[-1]}] deg."
        ) from error
    return airmass


def _validate_rectangular_field_region(
    interpolator: PsfMetricInterpolator, x: np.ndarray, y: np.ndarray
) -> None:
    validate_rectangular_field_coordinates(
        interpolator.x.to_value(u.arcsec),
        interpolator.y.to_value(u.arcsec),
        np.column_stack([x, y]),
        label="PSF metric",
        atol=_FIELD_ATOL,
    )


def _validate_point_or_line_region(
    source: np.ndarray, requested_coordinates: np.ndarray
) -> None:
    """Check distance to the fitted line and bounds along its sampled segment."""
    if source.shape[0] == 1:
        if not np.all(
            np.isclose(requested_coordinates, source[0], rtol=0.0, atol=_FIELD_ATOL)
        ):
            raise ValueError(
                "Requested field coordinates are outside the single-point source field region."
            )
        return

    center = np.mean(source, axis=0)
    _, singular_values, vh = np.linalg.svd(source - center, full_matrices=False)
    if singular_values[0] <= _FIELD_ATOL:
        raise ValueError("PSF metric source field region is degenerate.")
    direction = vh[0]
    source_projection = (source - center) @ direction
    requested_centered = requested_coordinates - center
    requested_projection = requested_centered @ direction
    perpendicular = requested_centered - np.outer(requested_projection, direction)
    if np.any(np.linalg.norm(perpendicular, axis=1) > _FIELD_ATOL):
        raise ValueError(
            "Requested field coordinates are outside the line source field region."
        )
    if np.any(requested_projection < np.min(source_projection) - _FIELD_ATOL) or np.any(
        requested_projection > np.max(source_projection) + _FIELD_ATOL
    ):
        raise ValueError(
            "Requested field coordinates are outside the line source field region."
        )


def _validate_sampled_field_region(
    interpolator: PsfMetricInterpolator, x: np.ndarray, y: np.ndarray
) -> None:
    """Check the sampled field region and handle line or point regions separately."""
    source = np.unique(field_coordinates(interpolator.x, interpolator.y), axis=0)
    requested_coordinates = np.column_stack([x, y])
    if (
        source.shape[0] < 3
        or np.linalg.matrix_rank(source - np.mean(source, axis=0), tol=_FIELD_ATOL) < 2
    ):
        _validate_point_or_line_region(source, requested_coordinates)
        return
    try:
        simplex = Delaunay(source).find_simplex(requested_coordinates, tol=_FIELD_ATOL)
    except QhullError:
        _validate_point_or_line_region(source, requested_coordinates)
        return
    if np.any(simplex < 0):
        raise ValueError(
            "Requested field coordinates are outside the source field region."
        )


def _requested_coordinate_values(
    interpolator: PsfMetricInterpolator, zenith_angle: Any, x: Any, y: Any
) -> dict[str, np.ndarray]:
    x = require_finite_vector(quantity_value(x, u.arcsec, label="x"), label="x")
    if x.size == 0:
        raise ValueError("At least one requested field coordinate is required.")
    y = require_finite_vector(
        quantity_value(y, u.arcsec, label="y"), label="y", length=x.size
    )
    precision = interpolator.grid_precision
    x = snap_coordinate_range(
        x,
        np.unique(interpolator.x.to_value(u.arcsec)),
        dtype=precision,
        label="field x",
    )
    y = snap_coordinate_range(
        y,
        np.unique(interpolator.y.to_value(u.arcsec)),
        dtype=precision,
        label="field y",
    )
    coordinate_values: dict[str, np.ndarray] = {COORD_X: x, COORD_Y: y}
    if COORD_AIRMASS in interpolator.coordinate_order:
        if zenith_angle is None:
            raise ValueError("zenith_angle is required by this PSF metric file.")
        zenith = require_finite_vector(
            quantity_value(zenith_angle, u.deg, label="zenith_angle"),
            label="zenith_angle",
        )
        if zenith.size == 1 and x.size != 1:
            zenith = np.full(x.shape, float(zenith[0]), dtype=float)
        if zenith.size != x.size:
            raise ValueError(
                "zenith_angle, x, and y must have matching lengths; "
                f"got {zenith.size}, {x.size}, {y.size}."
            )
        airmass = zenith_angle_to_airmass(zenith * u.deg).to_value(
            u.dimensionless_unscaled
        )
        airmass = _validate_airmass_range(interpolator, airmass, zenith)
        coordinate_values[COORD_AIRMASS] = airmass
    elif zenith_angle is not None and interpolator.zenith_angle_axis.size == 1:
        zenith = require_finite_vector(
            quantity_value(zenith_angle, u.deg, label="zenith_angle"),
            label="zenith_angle",
        )
        fixed_zenith = float(interpolator.zenith_angle_axis[0].to_value(u.deg))
        if not np.all(
            np.isclose(
                zenith,
                fixed_zenith,
                rtol=2 * np.finfo(precision).eps,
                atol=_FIELD_ATOL,
            )
        ):
            raise ValueError(
                "requested zenith_angle does not match this interpolator's "
                f"fixed zenith_angle={fixed_zenith}."
            )
    return coordinate_values


def _validated_coordinate_values(
    interpolator: PsfMetricInterpolator,
    *,
    zenith_angle: u.Quantity | None = None,
    x: u.Quantity,
    y: u.Quantity,
) -> dict[str, np.ndarray]:
    """Validate requested coordinates against the fitted field region and angle range."""
    if not isinstance(interpolator, PsfMetricInterpolator):
        raise TypeError("interpolator must be a PsfMetricInterpolator instance.")
    _validate_interpolator(interpolator)
    coordinate_values = _requested_coordinate_values(interpolator, zenith_angle, x, y)
    if interpolator.interpolation_type == PSF_METRIC_INTERPOLATION_TYPE_REGULAR_GRID:
        _validate_rectangular_field_region(
            interpolator, coordinate_values[COORD_X], coordinate_values[COORD_Y]
        )
    else:
        _validate_sampled_field_region(
            interpolator, coordinate_values[COORD_X], coordinate_values[COORD_Y]
        )
    return coordinate_values


# Model evaluation


def _evaluation_points(
    interpolator: PsfMetricInterpolator, coordinate_values: Mapping[str, np.ndarray]
) -> np.ndarray:
    return np.column_stack(
        [
            np.asarray(coordinate_values[name], dtype=interpolator.dtype)
            for name in interpolator.coordinate_order
        ]
    )


def _regular_grid_axes(interpolator: PsfMetricInterpolator) -> tuple[np.ndarray, ...]:
    axes: list[np.ndarray] = []
    for name in interpolator.coordinate_order:
        if name == COORD_AIRMASS:
            axes.append(interpolator.airmass_axis.to_value(u.dimensionless_unscaled))
        elif name == COORD_Y:
            axes.append(interpolator.y.to_value(u.arcsec))
        elif name == COORD_X:
            axes.append(interpolator.x.to_value(u.arcsec))
        else:
            raise ValueError(f"Unsupported PSF metric coordinate {name!r}.")
    return tuple(axes)


def _evaluate_grid_metrics(
    interpolator: PsfMetricInterpolator,
    *,
    coordinate_values: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    coordinates = field_coordinates(
        coordinate_values[COORD_X] * u.arcsec, coordinate_values[COORD_Y] * u.arcsec
    )
    coordinates = clip_rectangular_field_coordinates(
        interpolator.x.to_value(u.arcsec),
        interpolator.y.to_value(u.arcsec),
        coordinates,
    )
    coordinate_values = dict(coordinate_values)
    coordinate_values[COORD_X] = coordinates[:, 0]
    coordinate_values[COORD_Y] = coordinates[:, 1]
    points = _evaluation_points(interpolator, coordinate_values)
    axes = _regular_grid_axes(interpolator)
    return {
        name: np.asarray(
            RegularGridInterpolator(
                axes,
                quantity_value(
                    grid,
                    dict(interpolator._interpolation_data["metric_units"])[name],
                    label=f"metric grid {name}",
                ),
                method="linear",
                bounds_error=True,
            )(points),
            dtype=float,
        )
        for name, grid in dict(
            interpolator._interpolation_data.get("metric_grids", {})
        ).items()
    }


def _validate_predicted_metrics(metrics: Mapping[str, np.ndarray]) -> None:
    for name, values in metrics.items():
        values = np.asarray(values, dtype=float)
        if not np.all(np.isfinite(values)):
            raise ValueError(f"Interpolator produced non-finite {name} values.")
        if np.any(values <= 0.0):
            raise ValueError(f"Interpolator produced non-positive {name} values.")
        if name == "ee" and np.any(values > 1.0):
            raise ValueError("Interpolator produced ee values > 1.")


def _evaluate_psf_metric_interpolator(
    interpolator: PsfMetricInterpolator,
    *,
    zenith_angle: u.Quantity | None = None,
    x: u.Quantity,
    y: u.Quantity,
) -> PsfMetricPrediction:
    """Evaluate metrics at paired field coordinates within the fitted coordinate ranges.

    Args:
        interpolator: Built or loaded PSF metric interpolator.
        zenith_angle: Scalar angle or one angle per field point, compatible with
            degrees. Required for a varying airmass axis. A supplied angle
            must match any fixed zenith angle.
        x: Field x-coordinates compatible with arcseconds.
        y: Field y-coordinates compatible with arcseconds, paired with ``x``.

    Returns:
        Encircled energy and Strehl as dimensionless quantities, with
        FWHM in milliarcseconds, in requested point order.

    Raises:
        TypeError: If the file or provided coordinates have invalid types.
        ValueError: If units, shapes, coordinate ranges or predicted values are invalid.
    """

    coordinate_values = _validated_coordinate_values(
        interpolator,
        zenith_angle=zenith_angle,
        x=x,
        y=y,
    )
    if interpolator.interpolation_type == PSF_METRIC_INTERPOLATION_TYPE_RBF:
        coordinates = _evaluation_points(interpolator, coordinate_values)
        output = evaluate_scaled_rbfs(interpolator._interpolation_data, coordinates)
    else:
        output = _evaluate_grid_metrics(
            interpolator, coordinate_values=coordinate_values
        )
    _validate_predicted_metrics(output)
    return PsfMetricPrediction(
        dtype=interpolator.dtype,
        ee=np.asarray(output["ee"], dtype=interpolator.dtype)
        * u.dimensionless_unscaled,
        fwhm=np.asarray(output["fwhm"], dtype=interpolator.dtype) * u.mas,
        sr=np.asarray(output["sr"], dtype=interpolator.dtype)
        * u.dimensionless_unscaled,
    )


# File contents and validation


def _validate_metric_names(metric_names: tuple[str, ...], *, label: str) -> None:
    if len(set(metric_names)) != len(metric_names):
        raise ValueError(f"{label} contains duplicate metrics.")
    missing = set(REQUIRED_PSF_METRICS) - set(metric_names)
    if missing:
        raise ValueError(
            f"{label} is missing required metrics: {', '.join(sorted(missing))}."
        )
    unsupported = (
        set(metric_names) - set(REQUIRED_PSF_METRICS) - set(OPTIONAL_PSF_METRICS)
    )
    if unsupported:
        raise ValueError(
            f"{label} contains unsupported metrics: {', '.join(sorted(unsupported))}."
        )


def _validate_data_metric_names(
    expected_names: tuple[str, ...], data_names: set[str]
) -> None:
    metric_names = set(expected_names)
    if data_names != metric_names:
        missing = metric_names - data_names
        extra = data_names - metric_names
        details = []
        if missing:
            details.append(
                f"missing interpolation data metrics: {', '.join(sorted(missing))}"
            )
        if extra:
            details.append(
                f"extra interpolation data metrics: {', '.join(sorted(extra))}"
            )
        raise ValueError(
            f"PSF metric file interpolation data does not match metric_names ({'; '.join(details)})."
        )


def _validate_data_metric_units(
    interpolation_data: Mapping[str, Any],
) -> dict[str, u.UnitBase]:
    """Validate the fixed units carried by PSF metric interpolation data."""
    raw = interpolation_data.get("metric_units")
    if not isinstance(raw, Mapping) or set(raw) != set(_METRIC_UNITS):
        raise ValueError(
            "PSF metric file interpolation data must declare units for ee, fwhm, and sr."
        )
    for name, unit in _METRIC_UNITS.items():
        if u.Unit(raw[name]) != unit:
            raise ValueError(f"PSF metric file {name!r} unit must be {unit}.")
    return dict(_METRIC_UNITS)


def _expected_regular_grid_shape(
    interpolator: PsfMetricInterpolator,
) -> tuple[int, ...]:
    sizes = {
        COORD_AIRMASS: interpolator.airmass_axis.size,
        COORD_Y: interpolator.y.size,
        COORD_X: interpolator.x.size,
    }
    return tuple(int(sizes[name]) for name in interpolator.coordinate_order)


def _validate_rbf_interpolation_data(interpolator: PsfMetricInterpolator) -> None:
    interpolation_data = dict(interpolator._interpolation_data)
    _validate_data_metric_units(interpolation_data)
    data_names = set(dict(interpolation_data.get("evaluators", {})))
    _validate_data_metric_names(interpolator.metric_names, data_names)
    coordinate_mean = np.asarray(interpolation_data.get("coordinate_mean"), dtype=float)
    if coordinate_mean.shape != (len(interpolator.coordinate_order),):
        raise ValueError(
            "PSF metric RBF coordinate_mean shape must match coordinate_order."
        )


def _validate_grid_interpolation_data(interpolator: PsfMetricInterpolator) -> None:
    metric_grids = dict(interpolator._interpolation_data.get("metric_grids", {}))
    metric_units = _validate_data_metric_units(interpolator._interpolation_data)
    _validate_data_metric_names(interpolator.metric_names, set(metric_grids))
    expected_shape = _expected_regular_grid_shape(interpolator)
    for name, grid in metric_grids.items():
        grid = quantity_value(
            grid, metric_units[name], label=f"metric grid {name}", dtype=float
        )
        if grid.shape != expected_shape:
            raise ValueError(
                f"PSF metric file grid {name!r} must have shape {expected_shape}; got {grid.shape}."
            )
        if not np.all(np.isfinite(grid)):
            raise ValueError(
                f"PSF metric file grid {name!r} must contain only finite values."
            )


def _validate_interpolator(interpolator: PsfMetricInterpolator) -> None:
    """Check the common axes and validate the selected interpolation type's interpolation data."""
    if not isinstance(interpolator, PsfMetricInterpolator):
        raise TypeError("interpolator must be a PsfMetricInterpolator instance.")
    require_quantity(interpolator.zenith_angle_axis, u.deg, label="zenith_angle_axis")
    require_quantity(
        interpolator.airmass_axis, u.dimensionless_unscaled, label="airmass_axis"
    )
    require_quantity(interpolator.x, u.arcsec, label="x")
    require_quantity(interpolator.y, u.arcsec, label="y")
    if interpolator.airmass_axis.shape != interpolator.zenith_angle_axis.shape:
        raise ValueError("airmass_axis shape must match zenith_angle_axis shape.")
    _validate_metric_names(interpolator.metric_names, label="PSF metric file")
    interpolation_type = interpolator.interpolation_type
    _validate_coordinate_order_for_interpolation_type(
        tuple(interpolator.coordinate_order), interpolation_type
    )
    if COORD_AIRMASS in interpolator.coordinate_order:
        if interpolator.zenith_angle_axis.size <= 1:
            raise ValueError(
                "varying airmass coordinate requires multiple zenith_angle_axis values."
            )
    elif interpolator.zenith_angle_axis.size > 1:
        raise ValueError(
            "multiple zenith_angle_axis values require varying airmass coordinate."
        )
    if interpolation_type == PSF_METRIC_INTERPOLATION_TYPE_REGULAR_GRID:
        if interpolator.x.ndim != 1 or interpolator.y.ndim != 1:
            raise ValueError(
                "x and y must be 1-D rectangular field axes for regular-grid files."
            )
        if interpolator.x.size == 0 or interpolator.y.size == 0:
            raise ValueError("x and y field axes must not be empty.")
        if np.any(np.diff(interpolator.x) <= 0.0) or np.any(
            np.diff(interpolator.y) <= 0.0
        ):
            raise ValueError("x and y field axes must be strictly increasing.")
        _validate_grid_interpolation_data(interpolator)
        return
    if interpolator.x.shape != interpolator.y.shape:
        raise ValueError("x and y shapes must match.")
    if interpolator.x.size == 0:
        raise ValueError("x and y must not be empty.")
    _validate_rbf_interpolation_data(interpolator)


def _to_file_contents(interpolator: PsfMetricInterpolator) -> dict[str, Any]:
    """Build file contents with the interpolation type, interpolation data and units."""
    if not isinstance(interpolator, PsfMetricInterpolator):
        raise TypeError("interpolator must be a PsfMetricInterpolator instance.")
    _validate_interpolator(interpolator)
    return {
        "kind": PSF_METRIC_FILE_KIND,
        "version": PSF_METRIC_FILE_VERSION,
        "grid_precision": interpolator.grid_precision.name,
        "precision": interpolator.dtype.name,
        "interpolation_type": interpolator.interpolation_type,
        "interpolation": {"coordinate_order": tuple(interpolator.coordinate_order)},
        "metadata": {
            "zenith_angle_axis": interpolator.zenith_angle_axis,
            "airmass_axis": interpolator.airmass_axis,
            "x": interpolator.x,
            "y": interpolator.y,
            "metric_names": tuple(interpolator.metric_names),
        },
        "model": dict(interpolator._interpolation_data),
    }


def _from_file_contents(file_contents: Mapping[str, Any]) -> PsfMetricInterpolator:
    """Build a metric interpolator from checked file contents."""
    validate_file_format(
        file_contents,
        kind=PSF_METRIC_FILE_KIND,
        versions=(PSF_METRIC_FILE_VERSION,),
    )
    metadata = dict(file_contents.get("metadata", {}))
    interpolation_type = file_contents.get("interpolation_type")
    interpolation = dict(file_contents.get("interpolation", {}))
    coordinate_order = tuple(
        str(value) for value in interpolation.get("coordinate_order", ())
    )
    _validate_coordinate_order_for_interpolation_type(
        coordinate_order, interpolation_type
    )
    metric_names = tuple(str(value) for value in metadata.get("metric_names", ()))
    _validate_metric_names(metric_names, label="PSF metric file")
    interpolator = PsfMetricInterpolator(
        coordinate_order=coordinate_order,
        zenith_angle_axis=require_quantity(
            metadata.get("zenith_angle_axis"), u.deg, label="metadata.zenith_angle_axis"
        ),
        airmass_axis=require_quantity(
            metadata.get("airmass_axis"),
            u.dimensionless_unscaled,
            label="metadata.airmass_axis",
        ),
        x=require_quantity(metadata.get("x"), u.arcsec, label="metadata.x"),
        y=require_quantity(metadata.get("y"), u.arcsec, label="metadata.y"),
        metric_names=metric_names,
        interpolation_type=interpolation_type,
        _interpolation_data=dict(file_contents.get("model", {})),
        dtype=file_contents.get("precision", "float64"),
        grid_precision=file_contents["grid_precision"],
    )
    _validate_interpolator(interpolator)
    return interpolator


# Measure source images


def measure_psf_metrics(
    samples: PsfSamples,
    *,
    dtype: DTypeLike = np.float32,
    preserve_normalization: bool = False,
    fwhm_method: str = DEFAULT_FWHM_METHOD,
    fwhm_refinement_factor: int = DEFAULT_FWHM_REFINEMENT_FACTOR,
    ee_center: Literal["peak", "image"] = DEFAULT_EE_CENTER,
) -> PsfMetricSamples:
    """Measure FWHM, encircled energy and Strehl ratio from sampled PSF images.

    Encircled energy uses a circular aperture with radius equal to the
    measured FWHM.

    Args:
        samples: Source PSFs with their wavelength, scale and aperture metadata.
        dtype: Measurement and output precision, either float32 or float64.
        preserve_normalization: Measure the images as supplied when true.
            By default, AO Stats clips negative pixels and normalizes each
            crop to unit sum.
        fwhm_method: Width-estimation method passed to AO Stats.
        fwhm_refinement_factor: Subpixel sampling refinement passed to AO Stats.
        ee_center: Centre the energy aperture on the peak or image midpoint.

    Returns:
        Metric quantities with one row per input cube and one column per
        field point. FWHM is in milliarcseconds. Encircled energy and Strehl
        are dimensionless.

    Raises:
        TypeError: If the samples or physical metadata use unsupported types.
        ValueError: If the images, measurement options, units or precision
            are invalid.
    """

    dtype = validate_psf_dtype(dtype)
    samples = _validate_and_convert_measurement_samples(samples, dtype=dtype)
    sample_shape = samples.psfs.shape[:2]
    psfs = samples.psfs.reshape((-1, *samples.psfs.shape[2:]))
    metadata = _measurement_metadata(samples)
    preprocess = None if preserve_normalization else "default"

    sr = np.asarray(
        compute_psf_sr(psfs, metadata, preprocess=preprocess, dtype=dtype), dtype=dtype
    )
    fwhm = np.asarray(
        compute_psf_fwhm(
            psfs,
            metadata,
            fwhm_method=fwhm_method,
            fwhm_refinement_factor=fwhm_refinement_factor,
            preprocess=preprocess,
            dtype=dtype,
        ),
        dtype=dtype,
    )
    # Encircled energy uses a radius of one measured FWHM.
    ee_aperture_diameters = np.asarray(2.0 * fwhm, dtype=dtype).reshape(-1, 1) * u.mas
    ee = np.asarray(
        compute_psf_ee(
            psfs,
            metadata,
            ee_apertures=ee_aperture_diameters,
            ee_geometry=EE_GEOMETRY_ENCIRCLED,
            ee_center=ee_center,
            preprocess=preprocess,
            dtype=dtype,
        ),
        dtype=dtype,
    )
    return PsfMetricSamples(
        zenith_angle=samples.zenith_angle,
        x=samples.x,
        y=samples.y,
        ee=ee.reshape(sample_shape) * u.dimensionless_unscaled,
        fwhm=fwhm.reshape(sample_shape) * u.mas,
        sr=sr.reshape(sample_shape) * u.dimensionless_unscaled,
    )


# Build an interpolator


def build_psf_metric_interpolator(
    samples: PsfMetricSamples,
    *,
    dtype: DTypeLike = np.float32,
) -> PsfMetricInterpolator:
    """Fit a provider from supplied FWHM, encircled energy and Strehl samples.

    Complete rectangular field grids use linear interpolation. Scattered
    field points use a scaled thin-plate-spline RBF with smoothing 0.05 and a
    linear polynomial. Multiple unique zenith angles add an airmass coordinate.

    Args:
        samples: Measured metrics at each field point for each zenith angle.
        dtype: Evaluation precision, either float32 or float64.

    Returns:
        A read-only fitted metric provider ready to evaluate or save.

    Raises:
        TypeError: If samples or quantities have unsupported types.
        ValueError: If sampling, units, metric values or precision are invalid,
            including duplicate field points or zenith-angle samples.
    """

    dtype = validate_psf_dtype(dtype)
    source_samples = samples
    samples = _validate_and_convert_metric_samples(samples, dtype=dtype)
    zenith_axis = _zenith_axis(samples)
    if samples.zenith_angle is not None and zenith_axis.size != samples.ee.shape[0]:
        raise ValueError("PSF metric samples contain duplicate zenith-angle samples.")
    interpolation_type = _field_interpolation_type(samples)
    coordinate_order = _coordinate_order(interpolation_type, zenith_axis)
    if interpolation_type == PSF_METRIC_INTERPOLATION_TYPE_RBF:
        x, y = samples.x, samples.y
        interpolation_data = _fit_rbf_metrics(samples, coordinate_order)
    else:
        x, y, interpolation_data = _fit_regular_grid_metrics(
            samples, coordinate_order, zenith_axis
        )
    interpolation_data["metric_units"] = dict(_METRIC_UNITS)

    return PsfMetricInterpolator(
        dtype=dtype,
        coordinate_order=coordinate_order,
        zenith_angle_axis=zenith_axis * u.deg,
        airmass_axis=zenith_angle_to_airmass(zenith_axis * u.deg),
        x=x,
        y=y,
        metric_names=REQUIRED_PSF_METRICS,
        interpolation_type=interpolation_type,
        _interpolation_data=interpolation_data,
        grid_precision=coordinate_precision(
            dtype, source_samples.x, source_samples.y, source_samples.zenith_angle
        ),
    )._with_dtype(dtype)


def build_psf_metric_interpolator_from_psfs(
    samples: PsfSamples,
    *,
    dtype: DTypeLike = np.float32,
    preserve_normalization: bool = False,
) -> PsfMetricInterpolator:
    """Measure source images and select metric interpolation from their coordinates.

    Complete rectangular field grids use linear interpolation; scattered field
    points use the fixed RBF method described by build_psf_metric_interpolator.

    Args:
        samples: Source PSFs with their field, wavelength and aperture metadata.
        dtype: Measurement and evaluation precision, either float32 or float64.
        preserve_normalization: Use the images as supplied when true. The
            default clips negative pixels and normalizes each crop to unit sum.

    Returns:
        A fitted metric provider for encircled energy, FWHM and Strehl.

    Raises:
        TypeError: If samples or quantities have unsupported types.
        ValueError: If measurement or fitting rejects the samples or precision.
    """

    metrics = measure_psf_metrics(
        samples, dtype=dtype, preserve_normalization=preserve_normalization
    )
    return build_psf_metric_interpolator(
        metrics,
        dtype=dtype,
    )


# Compare reference samples


def compare_psf_metric_interpolator(
    interpolator: PsfMetricInterpolator,
    samples: PsfMetricSamples,
) -> PsfMetricComparisonSummary:
    """Evaluate at the supplied sample coordinates and compare metric values.

    Args:
        interpolator: Built or loaded PSF metric interpolator.
        samples: Reference metric values and coordinates for comparison.

    Returns:
        RMS and maximum absolute residual for each metric in its measurement
        units, together with the number of sampled zenith angles and field points.

    Raises:
        TypeError: If samples or quantities have invalid types.
        ValueError: If samples are invalid, lie outside the fitted coordinate ranges,
            or produce invalid metric predictions.
    """

    samples = _validate_and_convert_metric_samples(samples, dtype=interpolator.dtype)
    num_zenith_samples, num_points = samples.ee.shape
    zenith_angles = None
    if samples.zenith_angle is not None:
        zenith_angles = (
            np.repeat(samples.zenith_angle.to_value(u.deg), num_points) * u.deg
        )
    prediction = interpolator.evaluate(
        zenith_angle=zenith_angles,
        x=np.tile(samples.x.to_value(u.arcsec), num_zenith_samples) * u.arcsec,
        y=np.tile(samples.y.to_value(u.arcsec), num_zenith_samples) * u.arcsec,
    )
    residuals = {
        name: getattr(prediction, name).to_value(unit)
        - getattr(samples, name).to_value(unit).reshape(-1)
        for name, unit in _METRIC_UNITS.items()
    }
    return PsfMetricComparisonSummary(
        metric_rms_error={
            name: float(np.sqrt(np.mean(residual**2))) * _METRIC_UNITS[name]
            for name, residual in residuals.items()
        },
        metric_max_abs_error={
            name: float(np.max(np.abs(residual))) * _METRIC_UNITS[name]
            for name, residual in residuals.items()
        },
        num_planes=num_zenith_samples,
        num_points=num_points,
    )


# Save and load files


def save_psf_metric_interpolator(
    interpolator: PsfMetricInterpolator,
    path: Path,
    *,
    overwrite: bool = False,
) -> None:
    """Write a versioned pickle with the fitted metrics and their coordinate ranges.

    Create missing parent directories and publish the file atomically.

    Args:
        interpolator: Fitted metric provider to save.
        path: Destination pickle path.
        overwrite: Whether an existing file may be replaced.

    Raises:
        FileExistsError: If the destination exists and overwrite is false.
        TypeError: If the file type or metadata types are invalid.
        ValueError: If the file fails validation.
    """

    save_file_contents(_to_file_contents(interpolator), Path(path), overwrite=overwrite)


def load_psf_metric_interpolator(
    path: Path, *, dtype: DTypeLike | None = None
) -> PsfMetricInterpolator:
    """Load a fitted metric provider in its stored interpolation type.

    Pickle reconstruction can execute Python code. Load only trusted files.

    Args:
        path: Path to a saved PSF metric file.
        dtype: Optional float32 or float64 evaluation precision. The default
            preserves stored precision. Conversion retains coordinate rounding
            precision and does not change the saved file.

    Returns:
        A read-only metric provider at the selected precision.

    Raises:
        FileNotFoundError: If the file path does not exist.
        TypeError: If the file contents or metadata have invalid types.
        ValueError: If the file type, format version, interpolation type or interpolation data is invalid.
    """

    interpolator = _from_file_contents(load_file_contents(Path(path)))
    if dtype is not None and validate_psf_dtype(dtype) != interpolator.dtype:
        return interpolator._with_dtype(dtype)
    return interpolator
