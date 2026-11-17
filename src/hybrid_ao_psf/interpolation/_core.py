"""Shared interpolation file primitives."""

from __future__ import annotations

import pickle
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np
from astropy import units as u
from scipy.interpolate import RBFInterpolator

from .._io import atomic_output_path
from .._units import quantity_value

_RBF_KERNEL = "thin_plate_spline"
_RBF_SMOOTHING = 0.05
_RBF_POLYNOMIAL_DEGREE = 1
_AXIS_ATOL = 1.0e-10


class _ReadOnlyRbfInterpolator:
    """Owned, read-only callable wrapper around one fitted SciPy RBF evaluator."""

    __slots__ = ("__interpolator",)

    def __init__(self, interpolator: RBFInterpolator) -> None:
        object.__setattr__(
            self, "_ReadOnlyRbfInterpolator__interpolator", deepcopy(interpolator)
        )
        _make_object_arrays_read_only(self.__interpolator)

    def __call__(self, coordinates: np.ndarray) -> np.ndarray:
        return self.__interpolator(coordinates)

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError(
            f"{type(self).__name__} is read-only; cannot set {name!r}."
        )

    def __getstate__(self) -> RBFInterpolator:
        return deepcopy(self.__interpolator)

    def __setstate__(self, interpolator: RBFInterpolator) -> None:
        object.__setattr__(self, "_ReadOnlyRbfInterpolator__interpolator", interpolator)
        _make_object_arrays_read_only(self.__interpolator)


def _make_object_arrays_read_only(value: object) -> None:
    for item in vars(value).values():
        if isinstance(item, np.ndarray):
            item.flags.writeable = False


def make_read_only(value: Any) -> Any:
    """Return recursively owned, read-only file state where applicable."""
    if isinstance(value, u.Quantity):
        data = make_read_only(np.asarray(value))
        return u.Quantity(data, value.unit, dtype=data.dtype, copy=False)
    if isinstance(value, np.ndarray):
        owner = value
        while isinstance(owner, np.ndarray):
            owner = owner.base
        if isinstance(owner, bytes):
            return value.view()
        return np.frombuffer(value.tobytes(), dtype=value.dtype).reshape(value.shape)
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(name): make_read_only(item) for name, item in value.items()}
        )
    if isinstance(value, list | tuple):
        return tuple(make_read_only(item) for item in value)
    return value


def interpolator_views(value: Any) -> Any:
    """Share read-only buffers through views with independent metadata."""
    if isinstance(value, u.Quantity):
        return u.Quantity(np.asarray(value), value.unit, dtype=value.dtype, copy=False)
    if isinstance(value, np.ndarray):
        return value.view()
    if isinstance(value, Mapping):
        return MappingProxyType(
            {name: interpolator_views(item) for name, item in value.items()}
        )
    if isinstance(value, tuple):
        return tuple(interpolator_views(item) for item in value)
    return value


def cast_interpolator_value(value: Any, dtype: np.dtype) -> Any:
    """Copy owned physical/numerical state without altering nonfloating metadata."""
    if isinstance(value, u.Quantity):
        return u.Quantity(value.value, value.unit, dtype=dtype, copy=True)
    if isinstance(value, np.ndarray) and value.dtype.kind in "fc":
        target = (
            dtype
            if value.dtype.kind == "f"
            else (np.complex64 if dtype.itemsize == 4 else np.complex128)
        )
        return value.astype(target)
    if isinstance(value, Mapping):
        return {
            name: cast_interpolator_value(item, dtype) for name, item in value.items()
        }
    if isinstance(value, tuple):
        return tuple(cast_interpolator_value(item, dtype) for item in value)
    return value


def to_file_containers(value: Any) -> Any:
    """Convert protected mappings to ordinary containers for saving.

    Arrays and fitted evaluators are unchanged; this does not make data writable.
    """
    if isinstance(value, Mapping):
        return {str(name): to_file_containers(item) for name, item in value.items()}
    if isinstance(value, tuple):
        return tuple(to_file_containers(item) for item in value)
    return value


def zenith_angle_to_airmass(zenith_angle: u.Quantity) -> u.Quantity:
    """Convert zenith angle in degrees to airmass using ``sec(z)``.

    Args:
        zenith_angle: Scalar or array-like zenith angle values in degrees.

    Returns:
        A NumPy array of airmass values with the input shape.
    """

    values = quantity_value(zenith_angle, u.deg, label="zenith_angle")
    return (1.0 / np.cos(np.deg2rad(values))) * u.dimensionless_unscaled


def require_finite_vector(
    value: Any, *, label: str, length: int | None = None, dtype=None
) -> np.ndarray:
    """Return a finite 1-D float vector."""

    vector = np.asarray(value, dtype=dtype).reshape(-1)
    if length is not None and vector.size != int(length):
        raise ValueError(f"{label} must have length {int(length)}, got {vector.size}.")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{label} must contain only finite values.")
    return vector


def require_positive_vector(
    value: Any, *, label: str, length: int | None = None, dtype=None
) -> np.ndarray:
    """Return a finite positive 1-D float vector."""

    vector = require_finite_vector(value, label=label, length=length, dtype=dtype)
    if np.any(vector <= 0.0):
        raise ValueError(f"{label} must contain only values > 0.")
    return vector


def require_positive_scalar(value: Any, *, label: str) -> float:
    """Return a finite positive scalar float."""

    array = np.asarray(value, dtype=float)
    if array.ndim != 0:
        raise ValueError(f"{label} must be scalar.")
    scalar = float(array)
    if not np.isfinite(scalar) or scalar <= 0.0:
        raise ValueError(f"{label} must be finite and > 0.")
    return scalar


def require_pupil(
    value: Any, *, label: str = "tel_pupil", dtype=np.float32
) -> np.ndarray:
    """Return a finite 2-D telescope pupil array."""

    pupil = np.asarray(value, dtype=dtype)
    if pupil.ndim != 2:
        raise ValueError(f"{label} must be a 2-D array; got shape {pupil.shape}.")
    if not np.all(np.isfinite(pupil)):
        raise ValueError(f"{label} must contain only finite values.")
    return pupil


def normalize_psfs(psfs: np.ndarray) -> np.ndarray:
    """Clip negative PSF pixels and normalize each PSF to unit sum."""

    psf_cube = np.maximum(np.asarray(psfs, dtype=np.float32), 0.0)
    totals = np.sum(psf_cube, axis=(-2, -1), dtype=np.float32, keepdims=True)
    if np.any(totals <= 0.0):
        raise ValueError("Cannot normalize PSFs with non-positive total flux.")
    return (psf_cube / totals).astype(np.float32, copy=False)


def validate_psf_array(
    psfs: Any, *, label: str, ndim: int, dtype=np.float32
) -> np.ndarray:
    """Return a finite PSF array with the expected dimensionality."""

    array = np.asarray(psfs, dtype=dtype)
    if array.ndim != ndim:
        raise ValueError(f"{label} must have ndim={ndim}, got shape {array.shape}.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{label} must contain only finite values.")
    if np.any(array < 0.0):
        raise ValueError(f"{label} must not contain negative values.")
    return array


def field_coordinates(x: Any, y: Any) -> np.ndarray:
    """Return validated ``(x, y)`` field coordinates."""

    x = require_finite_vector(quantity_value(x, u.arcsec, label="x"), label="x")
    y = require_finite_vector(
        quantity_value(y, u.arcsec, label="y"), label="y", length=x.size
    )
    if x.size == 0:
        raise ValueError("At least one field coordinate is required.")
    return np.column_stack([x, y])


def rectangular_field_axes(
    x: Any, y: Any, *, label: str
) -> tuple[np.ndarray, np.ndarray]:
    """Return sorted rectangular field axes after validating a complete field grid."""

    coordinates = field_coordinates(x, y)
    x_axis = unique_sorted(coordinates[:, 0], label="x")
    y_axis = unique_sorted(coordinates[:, 1], label="y")
    if coordinates.shape[0] != x_axis.size * y_axis.size:
        raise ValueError(
            f"{label} must form a complete rectangular x x y field grid; "
            f"got {coordinates.shape[0]} points for {x_axis.size} x {y_axis.size} axes."
        )
    seen: set[tuple[int, int]] = set()
    for x_value, y_value in coordinates:
        ix = axis_index(x_axis, float(x_value), label="x")
        iy = axis_index(y_axis, float(y_value), label="y")
        key = (iy, ix)
        if key in seen:
            raise ValueError(
                f"Duplicate {label} field point at x={x_value}, y={y_value}."
            )
        seen.add(key)
    return x_axis, y_axis


def grid_field_values(
    x: Any,
    y: Any,
    values: Any,
    x_axis: np.ndarray,
    y_axis: np.ndarray,
    *,
    label: str,
    dtype: Any,
) -> np.ndarray:
    """Place point-indexed values into a rectangular ``(y, x, ...)`` grid."""

    coordinates = field_coordinates(x, y)
    values = np.asarray(values, dtype=dtype)
    if values.shape[0] != coordinates.shape[0]:
        raise ValueError(
            f"{label} values must have leading dimension {coordinates.shape[0]}; got {values.shape[0]}."
        )
    grid = np.full((y_axis.size, x_axis.size, *values.shape[1:]), np.nan, dtype=dtype)
    for point_index, (x_value, y_value) in enumerate(coordinates):
        ix = axis_index(x_axis, float(x_value), label="x")
        iy = axis_index(y_axis, float(y_value), label="y")
        grid[iy, ix] = values[point_index]
    if not np.all(np.isfinite(grid)):
        raise ValueError(f"{label} field grid is incomplete.")
    return grid


def validate_rectangular_field_coordinates(
    x_axis: Any,
    y_axis: Any,
    coordinates: np.ndarray,
    *,
    label: str,
    atol: float,
) -> None:
    """Validate requested coordinates against rectangular field-axis bounds."""

    x_axis = require_finite_vector(x_axis, label="x_axis")
    y_axis = require_finite_vector(y_axis, label="y_axis")
    coordinates = np.asarray(coordinates, dtype=float)
    x = np.asarray(coordinates[:, 0], dtype=float)
    y = np.asarray(coordinates[:, 1], dtype=float)
    if np.any(x < x_axis[0] - atol) or np.any(x > x_axis[-1] + atol):
        raise ValueError(
            f"requested x coordinate is outside {label} x range [{x_axis[0]}, {x_axis[-1]}]."
        )
    if np.any(y < y_axis[0] - atol) or np.any(y > y_axis[-1] + atol):
        raise ValueError(
            f"requested y coordinate is outside {label} y range [{y_axis[0]}, {y_axis[-1]}]."
        )


def clip_rectangular_field_coordinates(
    x_axis: Any, y_axis: Any, coordinates: np.ndarray
) -> np.ndarray:
    """Clip requested coordinates to the rectangular field bounds."""

    clipped = np.asarray(coordinates, dtype=float).copy()
    x_axis = require_finite_vector(x_axis, label="x_axis")
    y_axis = require_finite_vector(y_axis, label="y_axis")
    clipped[:, 0] = np.clip(clipped[:, 0], x_axis[0], x_axis[-1])
    clipped[:, 1] = np.clip(clipped[:, 1], y_axis[0], y_axis[-1])
    return clipped


def fit_scaled_rbfs(
    coordinates: np.ndarray,
    values_by_name: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    """Fit scaled RBF evaluators and return their private interpolation data."""

    dtype = np.asarray(next(iter(values_by_name.values()))).dtype
    coordinates = np.asarray(coordinates, dtype=dtype)
    if coordinates.ndim != 2 or coordinates.shape[0] == 0:
        raise ValueError("coordinates must be a non-empty 2-D array.")
    if not np.all(np.isfinite(coordinates)):
        raise ValueError("coordinates must contain only finite values.")

    coordinate_mean = np.nanmean(coordinates, axis=0)
    coordinate_scale = np.nanstd(coordinates, axis=0)
    varying_axes = coordinate_scale > 0.0
    if not np.any(varying_axes):
        raise ValueError("At least one interpolation coordinate dimension must vary.")

    varying_coordinates = coordinates[:, varying_axes]
    varying_mean = coordinate_mean[varying_axes]
    varying_scale = coordinate_scale[varying_axes]
    scaled_coordinates = (varying_coordinates - varying_mean) / varying_scale

    evaluators: dict[str, _ReadOnlyRbfInterpolator] = {}
    for name, values in values_by_name.items():
        values = np.asarray(values, dtype=dtype)
        if values.shape[0] != coordinates.shape[0]:
            raise ValueError(
                f"{name} has leading dimension {values.shape[0]}; expected {coordinates.shape[0]}."
            )
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{name} contains non-finite values.")
        evaluators[str(name)] = _ReadOnlyRbfInterpolator(
            RBFInterpolator(
                scaled_coordinates,
                values,
                kernel=_RBF_KERNEL,
                smoothing=_RBF_SMOOTHING,
                degree=_RBF_POLYNOMIAL_DEGREE,
            )
        )

    return {
        "coordinate_mean": coordinate_mean,
        "coordinate_scale": coordinate_scale,
        "varying_axes": varying_axes,
        "evaluators": evaluators,
    }


def evaluate_scaled_rbfs(
    interpolation_data: Mapping[str, Any], coordinates: np.ndarray
) -> dict[str, np.ndarray]:
    """Evaluate the scaled RBFs using their private interpolation data."""

    dtype = np.asarray(interpolation_data["coordinate_mean"]).dtype
    coordinates = np.asarray(coordinates, dtype=dtype)
    varying_axes = np.asarray(interpolation_data["varying_axes"], dtype=bool)
    coordinate_mean = np.asarray(interpolation_data["coordinate_mean"], dtype=dtype)
    coordinate_scale = np.asarray(interpolation_data["coordinate_scale"], dtype=dtype)
    scaled = (
        coordinates[:, varying_axes] - coordinate_mean[varying_axes]
    ) / coordinate_scale[varying_axes]
    return {
        name: np.asarray(interpolator(scaled), dtype=dtype)
        for name, interpolator in dict(interpolation_data["evaluators"]).items()
    }


def unique_sorted(values: np.ndarray, *, label: str) -> np.ndarray:
    """Return unique sorted axis values with duplicate detection."""

    values = require_finite_vector(values, label=label)
    if values.size == 0:
        raise ValueError(f"{label} must not be empty.")
    # Ten decimal places are below float32 resolution; scaling by 1e10
    # can itself move representable sample points such as 30 or 60.
    rounded = (
        values if values.dtype == np.dtype("float32") else np.round(values, decimals=10)
    )
    unique = np.asarray(sorted({float(value) for value in rounded}), dtype=float)
    return unique


def axis_index(axis: np.ndarray, value: float, *, label: str) -> int:
    """Return the unique index matching ``value`` on ``axis``."""

    matches = np.where(np.isclose(axis, float(value), rtol=0.0, atol=_AXIS_ATOL))[0]
    if matches.size != 1:
        raise ValueError(f"Could not locate {label} value {value}.")
    return int(matches[0])


def interpolation_axis_weights(
    axis: np.ndarray, value: float, *, label: str
) -> list[tuple[int, float]]:
    """Return linear interpolation weights, rejecting out-of-range values."""

    axis = np.asarray(axis).reshape(-1)
    value = axis.dtype.type(value)
    if axis.size == 0:
        raise ValueError(f"{label} axis is empty.")
    if axis.size == 1:
        if np.isclose(value, axis[0], rtol=0.0, atol=_AXIS_ATOL):
            return [(0, 1.0)]
        raise ValueError(
            f"{label}={value} is outside the sampled range [{axis[0]}, {axis[0]}]."
        )
    if value < axis[0] and not np.isclose(value, axis[0], rtol=0.0, atol=_AXIS_ATOL):
        raise ValueError(
            f"{label}={value} is below the sampled range minimum {axis[0]}."
        )
    if value > axis[-1] and not np.isclose(value, axis[-1], rtol=0.0, atol=_AXIS_ATOL):
        raise ValueError(
            f"{label}={value} is above the sampled range maximum {axis[-1]}."
        )
    if np.isclose(value, axis[0], rtol=0.0, atol=_AXIS_ATOL):
        return [(0, 1.0)]
    if np.isclose(value, axis[-1], rtol=0.0, atol=_AXIS_ATOL):
        return [(axis.size - 1, 1.0)]
    upper = int(np.searchsorted(axis, value, side="right"))
    lower = upper - 1
    if np.isclose(axis[lower], value, rtol=0.0, atol=_AXIS_ATOL):
        return [(lower, 1.0)]
    if np.isclose(axis[upper], value, rtol=0.0, atol=_AXIS_ATOL):
        return [(upper, 1.0)]
    weight = float((value - axis[lower]) / (axis[upper] - axis[lower]))
    return [(lower, 1.0 - weight), (upper, weight)]


def save_file_contents(
    file_contents: Mapping[str, Any], path: Path, *, overwrite: bool
) -> None:
    """Save file contents to a pickle, replacing the destination atomically."""

    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing file: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with (
        atomic_output_path(path) as temporary_path,
        temporary_path.open("wb") as handle,
    ):
        pickle.dump(
            to_file_containers(file_contents),
            handle,
            protocol=pickle.HIGHEST_PROTOCOL,
        )


def load_file_contents(path: Path) -> dict[str, Any]:
    """Load the contents of a trusted pickle and require a dictionary."""

    with Path(path).open("rb") as handle:
        file_contents = pickle.load(handle)
    if not isinstance(file_contents, dict):
        raise TypeError(
            f"{path} does not contain a Hybrid AO PSF interpolation dictionary."
        )
    return file_contents


def validate_file_format(
    file_contents: Mapping[str, Any], *, kind: str, versions: tuple[int, ...]
) -> None:
    """Check the saved file type and supported format versions."""

    if file_contents.get("kind") != kind:
        raise ValueError(f"Unsupported file kind: {file_contents.get('kind')!r}.")
    if file_contents.get("version") not in versions:
        raise ValueError(f"Unsupported file version: {file_contents.get('version')!r}.")


def file_grid_precision(file_contents: Mapping[str, Any]) -> str:
    """Read the required coordinate precision from a current interpolator file."""
    grid_precision = file_contents.get("grid_precision")
    if not isinstance(grid_precision, str):
        raise TypeError("Interpolator file requires a grid_precision string.")
    return grid_precision
