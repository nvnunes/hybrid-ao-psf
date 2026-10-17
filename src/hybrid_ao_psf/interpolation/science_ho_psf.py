"""Science high-order PSF interpolation artifacts."""

from __future__ import annotations

from collections.abc import Mapping
from copy import copy
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
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

from .._units import quantity_value, require_quantity
from .._validation import (
    coordinate_support_dtype,
    support_dtype,
    validate_meta_field_name,
    validate_psf_dtype,
)
from ..contracts import ScienceHoPsfPrediction
from ._core import (
    axis_index,
    cast_artifact_value,
    field_coordinates,
    freeze_artifact_value,
    grid_field_values,
    interpolation_axis_weights,
    load_payload,
    rectangular_field_axes,
    require_finite_vector,
    require_positive_scalar,
    require_positive_vector,
    require_pupil,
    save_payload,
    snap_rectangular_field_query,
    unique_sorted,
    validate_payload_kind,
    validate_rectangular_field_query,
    zenith_angle_to_airmass,
)

SCIENCE_HO_PSF_ARTIFACT_KIND = "hybrid_ao_psf_science_ho_psf_interpolator"
SCIENCE_HO_PSF_ARTIFACT_VERSION = 2
SCIENCE_HO_PSF_FIELD_QUERY_ATOL = 1.0e-6 * u.arcsec
SCIENCE_COORD_AIRMASS = "airmass"
SCIENCE_COORD_WAVELENGTH = "wavelength"
SCIENCE_COORD_Y = "y"
SCIENCE_COORD_X = "x"
SCIENCE_FIELD_COORDINATE_ORDER = (SCIENCE_COORD_Y, SCIENCE_COORD_X)
SCIENCE_VALID_COORDINATE_ORDERS = frozenset(
    {
        SCIENCE_FIELD_COORDINATE_ORDER,
        (SCIENCE_COORD_WAVELENGTH, *SCIENCE_FIELD_COORDINATE_ORDER),
        (SCIENCE_COORD_AIRMASS, *SCIENCE_FIELD_COORDINATE_ORDER),
        (
            SCIENCE_COORD_AIRMASS,
            SCIENCE_COORD_WAVELENGTH,
            *SCIENCE_FIELD_COORDINATE_ORDER,
        ),
    }
)
_AXIS_ATOL = 1.0e-10


@dataclass(frozen=True, kw_only=True)
class ScienceHoPsfSamples:
    """Science high-order PSF samples used to build an interpolation artifact.

    ``wavelength`` is required physical PSF metadata. It becomes an active
    interpolation coordinate only when more than one wavelength is supplied.
    ``zenith_angle`` is optional; when omitted, the artifact has no zenith
    or airmass support check. When supplied with one unique value, it is stored
    as fixed metadata rather than as an active interpolation coordinate.

    Attributes:
        x: Science field x-coordinates in arcseconds.
        y: Science field y-coordinates in arcseconds.
        psfs: PSF array with shape ``(points, y, x)`` for fixed-plane samples
            or ``(planes, points, y, x)`` for physical-axis samples. Columns
            increase toward ``+x`` and rows toward ``+y`` in the field frame;
            source orientation is supplied by the caller, not inferred from
            the PSF metadata.
        wavelength: Scalar fixed wavelength or one wavelength per plane.
        pixel_scale: Scalar fixed pixel scale or one value per plane.
        tel_diameter: Telescope diameter in meters.
        tel_pupil: Shared two-dimensional telescope pupil.
        zenith_angle: Optional scalar fixed zenith angle or one value per
            plane. Multiple unique values activate airmass interpolation.
        meta: Optional source metadata quantities. Values may be
            artifact-global finite scalar quantities or finite quantity vectors
            with one value per source plane.
        provenance: Optional source provenance strings.
    """

    x: u.Quantity
    y: u.Quantity
    psfs: np.ndarray
    wavelength: u.Quantity
    pixel_scale: u.Quantity
    tel_diameter: u.Quantity
    tel_pupil: u.Quantity
    zenith_angle: u.Quantity | None = None
    meta: Mapping[str, Any] = field(default_factory=dict)
    provenance: tuple[str, ...] = ()


@dataclass(frozen=True)
class ScienceHoPsfReplaySummary:
    """Source-node replay residual summary for a science-HO-PSF artifact."""

    psf_nrms_mean: float
    psf_nrms_max: float
    pixel_scale_abs_max: u.Quantity
    metric_rms: dict[str, u.Quantity]
    metric_max_abs: dict[str, u.Quantity]
    num_planes: int
    num_points: int


@dataclass(frozen=True)
class ScienceHoPsfInterpolator:
    """Versioned science-HO-PSF interpolation artifact.

    ``coordinate_order`` records the active interpolation coordinates. Fixed
    wavelength and fixed zenith values remain in their metadata axes but are not
    included in ``coordinate_order``. The persisted provider contains a
    prepared SciPy interpolator and is ready to evaluate after loading.
    ``dtype`` selects float32 pixels by default or explicit float64; storage
    and predictions preserve this precision without changing source flux.
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
    meta: Mapping[str, Any] = field(default_factory=dict)
    provenance: tuple[str, ...] = ()
    builder: Mapping[str, Any] = field(default_factory=dict)
    dtype: DTypeLike = field(default=np.float32, kw_only=True)
    _prepared_psf_interpolator: RegularGridInterpolator | None = field(
        default=None, init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "coordinate_order", tuple(self.coordinate_order))
        object.__setattr__(
            self, "zenith_angle_axis", freeze_artifact_value(self.zenith_angle_axis)
        )
        object.__setattr__(
            self, "airmass_axis", freeze_artifact_value(self.airmass_axis)
        )
        object.__setattr__(
            self, "wavelength_axis", freeze_artifact_value(self.wavelength_axis)
        )
        object.__setattr__(self, "x", freeze_artifact_value(self.x))
        object.__setattr__(self, "y", freeze_artifact_value(self.y))
        object.__setattr__(
            self, "psf_shape", tuple(int(value) for value in self.psf_shape)
        )
        object.__setattr__(
            self, "pixel_scale_grid", freeze_artifact_value(self.pixel_scale_grid)
        )
        object.__setattr__(
            self, "tel_diameter", freeze_artifact_value(self.tel_diameter)
        )
        object.__setattr__(self, "tel_pupil", freeze_artifact_value(self.tel_pupil))
        dtype = validate_psf_dtype(self.dtype)
        object.__setattr__(self, "dtype", dtype)
        object.__setattr__(
            self,
            "psf_grid",
            freeze_artifact_value(np.asarray(self.psf_grid, dtype=dtype)),
        )
        object.__setattr__(self, "meta", freeze_artifact_value(self.meta))
        object.__setattr__(
            self, "provenance", tuple(str(value) for value in self.provenance)
        )
        object.__setattr__(self, "builder", freeze_artifact_value(self.builder))

    def prepare(self, *, dtype: DTypeLike = np.float32) -> ScienceHoPsfInterpolator:
        """Return owned runtime state in dtype without changing this artifact.

        Builder provenance retains the original producer precision. Promotion
        cannot recover information rounded during calibration.
        """
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
            "meta",
        )
        return replace(
            self,
            dtype=dtype,
            builder={
                **self.builder,
                "support_precision": support_dtype(
                    self.dtype,
                    dtype,
                    self.builder.get(
                        "support_precision", self.builder.get("precision")
                    ),
                ).name,
            },
            **{name: cast_artifact_value(getattr(self, name), dtype) for name in names},
        )

    def predict(
        self,
        *,
        zenith_angle: u.Quantity,
        wavelength: u.Quantity,
        x: u.Quantity,
        y: u.Quantity,
    ) -> ScienceHoPsfPrediction:
        """Evaluate this loaded provider at one request's coordinates."""
        return evaluate_science_ho_psf_interpolator(
            self,
            zenith_angle=zenith_angle,
            wavelength=wavelength,
            x=x,
            y=y,
        )

    def _prepared_runtime(self) -> _ScienceHoPsfRuntimeInterpolator:
        psf_interpolator = self._prepared_psf_interpolator
        if psf_interpolator is None:
            runtime = _prepare_science_ho_psf_runtime_interpolator(self)
            psf_interpolator = runtime.psf_interpolator
            object.__setattr__(self, "_prepared_psf_interpolator", psf_interpolator)
        return _ScienceHoPsfRuntimeInterpolator(self, psf_interpolator)


@dataclass(frozen=True)
class _ScienceHoPsfRuntimeInterpolator:
    """Process-local prepared interpolator for a science-HO-PSF artifact."""

    artifact: ScienceHoPsfInterpolator
    psf_interpolator: RegularGridInterpolator


def build_science_ho_psf_interpolator(
    samples: ScienceHoPsfSamples,
    *,
    dtype: DTypeLike = np.float32,
) -> ScienceHoPsfInterpolator:
    """Build a science high-order PSF interpolation artifact.

    Active physical axes are inferred from supplied sample diversity. Field
    coordinates must form a complete rectangular ``y x x`` grid. ``dtype``
    accepts float32 (the default) or float64 and applies to stored and returned
    pixels. Source flux and image orientation are preserved in either case.
    """

    dtype = validate_psf_dtype(dtype)
    prepared = _prepare_samples(samples, dtype=dtype)
    field_x_axis, field_y_axis = rectangular_field_axes(
        prepared.x,
        prepared.y,
        label="Science HO PSF samples",
    )
    wavelength_axis = unique_sorted(
        prepared.wavelength.to_value(u.um), label="wavelength"
    )
    zenith_axis = (
        unique_sorted(prepared.zenith_angle.to_value(u.deg), label="zenith_angle")
        if prepared.zenith_angle is not None
        else np.asarray([], dtype=dtype)
    )
    coordinate_order = _coordinate_order_for_axes(zenith_axis, wavelength_axis)
    active_shape = _active_physical_shape(
        coordinate_order, zenith_axis, wavelength_axis
    )
    plane_indices = np.full(active_shape, -1, dtype=int)
    psf_grid = np.full(
        (*active_shape, field_y_axis.size, field_x_axis.size, *prepared.psfs.shape[2:]),
        np.nan,
        dtype=dtype,
    )

    for plane_index in range(prepared.psfs.shape[0]):
        grid_index = _active_grid_index(
            coordinate_order,
            zenith_axis,
            wavelength_axis,
            None
            if prepared.zenith_angle is None
            else float(prepared.zenith_angle[plane_index].to_value(u.deg)),
            float(prepared.wavelength[plane_index].to_value(u.um)),
        )
        if plane_indices[grid_index] != -1:
            raise ValueError(
                "Duplicate science HO PSF plane for active physical coordinates."
            )
        plane_indices[grid_index] = plane_index
        psf_grid[grid_index] = grid_field_values(
            prepared.x,
            prepared.y,
            prepared.psfs[plane_index],
            field_x_axis,
            field_y_axis,
            label="Science HO PSF sample",
            dtype=dtype,
        )

    if np.any(plane_indices < 0):
        raise ValueError(
            "Science HO PSF samples must form a complete active physical-coordinate grid."
        )

    pixel_scale_grid = np.full(active_shape, np.nan, dtype=dtype)
    for plane_index, pixel_scale in enumerate(prepared.pixel_scale.to_value(u.mas)):
        grid_index = _active_grid_index(
            coordinate_order,
            zenith_axis,
            wavelength_axis,
            None
            if prepared.zenith_angle is None
            else float(prepared.zenith_angle[plane_index].to_value(u.deg)),
            float(prepared.wavelength[plane_index].to_value(u.um)),
        )
        pixel_scale_grid[grid_index] = float(pixel_scale)
    meta = _build_artifact_meta(
        prepared.meta,
        prepared,
        coordinate_order,
        zenith_axis,
        wavelength_axis,
        active_shape,
        dtype=dtype,
    )

    return ScienceHoPsfInterpolator(
        coordinate_order=coordinate_order,
        zenith_angle_axis=zenith_axis * u.deg,
        airmass_axis=zenith_angle_to_airmass(zenith_axis * u.deg),
        wavelength_axis=wavelength_axis * u.um,
        x=np.asarray(field_x_axis, dtype=dtype) * u.arcsec,
        y=np.asarray(field_y_axis, dtype=dtype) * u.arcsec,
        psf_shape=tuple(int(v) for v in prepared.psfs.shape[2:]),
        pixel_scale_grid=np.asarray(pixel_scale_grid, dtype=dtype) * u.mas,
        tel_diameter=prepared.tel_diameter,
        tel_pupil=prepared.tel_pupil,
        meta=meta,
        psf_grid=psf_grid,
        dtype=dtype,
        provenance=tuple(prepared.provenance),
        builder={
            "name": "hybrid_ao_psf.interpolation.science_ho_psf",
            "precision": dtype.name,
            "support_precision": coordinate_support_dtype(
                dtype, samples.x, samples.y, samples.zenith_angle, samples.wavelength
            ).name,
            "created_at": datetime.now(UTC).isoformat(),
            "interpolation_method": _interpolation_method(coordinate_order),
        },
    ).prepare(dtype=dtype)


def _prepare_science_ho_psf_runtime_interpolator(
    interpolator: ScienceHoPsfInterpolator,
) -> _ScienceHoPsfRuntimeInterpolator:
    """Prepare a runtime interpolator for repeated science-HO-PSF queries."""

    validate_science_ho_psf_interpolator(interpolator)
    return _ScienceHoPsfRuntimeInterpolator(
        artifact=interpolator,
        psf_interpolator=RegularGridInterpolator(
            _interpolation_axes(interpolator),
            np.asarray(interpolator.psf_grid),
            method="linear",
            bounds_error=True,
        ),
    )


def _evaluate_science_ho_psf_runtime_interpolator(
    runtime_interpolator: _ScienceHoPsfRuntimeInterpolator,
    *,
    zenith_angle: u.Quantity | None = None,
    wavelength: u.Quantity | None = None,
    x: u.Quantity,
    y: u.Quantity,
) -> ScienceHoPsfPrediction:
    """Evaluate a prepared runtime science-HO-PSF interpolator."""

    if not isinstance(runtime_interpolator, _ScienceHoPsfRuntimeInterpolator):
        raise TypeError(
            "runtime_interpolator must be a _ScienceHoPsfRuntimeInterpolator instance."
        )
    interpolator = runtime_interpolator.artifact
    coordinates = field_coordinates(x, y)
    _validate_field_query(interpolator, coordinates)
    coordinates = _snap_field_query(interpolator, coordinates)
    physical_values = _validate_and_prepare_query_physical_values(
        interpolator,
        zenith_angle=zenith_angle,
        wavelength=wavelength,
    )
    points = _query_points(interpolator, physical_values, coordinates)
    psfs = np.asarray(
        runtime_interpolator.psf_interpolator(points), dtype=interpolator.dtype
    ).reshape(
        coordinates.shape[0],
        *interpolator.psf_shape,
    )
    weights = _plane_weights(interpolator, physical_values)
    pixel_scale = 0.0
    meta = _initial_evaluated_meta(
        interpolator.meta,
        grid_shape=_active_grid_shape(interpolator),
        dtype=interpolator.dtype,
    )
    for weight, grid_index in weights:
        pixel_scale += float(weight) * float(
            interpolator.pixel_scale_grid.to_value(u.mas)[grid_index]
        )
        _accumulate_grid_meta(
            meta, interpolator.meta, weight=float(weight), grid_index=grid_index
        )
    _validate_psf_flux(psfs, label="predicted psfs")
    wavelength_value = _prediction_wavelength(interpolator, physical_values)
    pixel_scale = float(pixel_scale)
    return ScienceHoPsfPrediction(
        psfs=psfs,
        dtype=interpolator.dtype,
        pixel_scale=pixel_scale * u.mas,
        metadata=PsfMetadata(
            wavelength=wavelength_value * u.um,
            pixel_scale=pixel_scale * u.mas,
            tel_diameter=interpolator.tel_diameter,
            tel_pupil=interpolator.tel_pupil,
        ),
        meta=meta,
    )


def evaluate_science_ho_psf_interpolator(
    interpolator: ScienceHoPsfInterpolator,
    *,
    zenith_angle: u.Quantity | None = None,
    wavelength: u.Quantity | None = None,
    x: u.Quantity,
    y: u.Quantity,
) -> ScienceHoPsfPrediction:
    """Evaluate a science-HO-PSF artifact at supported coordinates.

    ``zenith_angle`` is required only when the artifact has an active
    ``airmass`` coordinate. ``wavelength`` is required only when wavelength
    is active; otherwise a supplied value is checked against the fixed artifact
    wavelength.
    """
    if not isinstance(interpolator, ScienceHoPsfInterpolator):
        raise TypeError("interpolator must be a ScienceHoPsfInterpolator instance.")

    return _evaluate_science_ho_psf_runtime_interpolator(
        interpolator._prepared_runtime(),
        zenith_angle=zenith_angle,
        wavelength=wavelength,
        x=x,
        y=y,
    )


def validate_science_ho_psf_query(
    interpolator: ScienceHoPsfInterpolator,
    *,
    zenith_angle: u.Quantity | None = None,
    wavelength: u.Quantity | None = None,
) -> None:
    """Validate science-HO-PSF physical-coordinate support without evaluating PSFs."""

    if not isinstance(interpolator, ScienceHoPsfInterpolator):
        raise TypeError("interpolator must be a ScienceHoPsfInterpolator instance.")
    _validate_interpolator(interpolator)
    physical_values = _validate_and_prepare_query_physical_values(
        interpolator,
        zenith_angle=zenith_angle,
        wavelength=wavelength,
    )
    _plane_weights(interpolator, physical_values)


def validate_science_ho_psf_interpolator(
    interpolator: ScienceHoPsfInterpolator,
) -> None:
    """Validate a science-HO-PSF artifact contract."""

    if not isinstance(interpolator, ScienceHoPsfInterpolator):
        raise TypeError("interpolator must be a ScienceHoPsfInterpolator instance.")
    _validate_interpolator(interpolator)


def replay_science_ho_psf_interpolator(
    interpolator: ScienceHoPsfInterpolator,
    samples: ScienceHoPsfSamples,
) -> ScienceHoPsfReplaySummary:
    """Replay a science-HO-PSF artifact at source sample nodes."""

    prepared = _prepare_samples(samples, dtype=interpolator.dtype)
    residuals: list[np.ndarray] = []
    pixel_scale_errors: list[float] = []
    metric_errors: dict[str, list[np.ndarray]] = {"fwhm": [], "ee": []}
    for index in range(prepared.psfs.shape[0]):
        wavelength = prepared.wavelength[index]
        prediction = evaluate_science_ho_psf_interpolator(
            interpolator,
            zenith_angle=None
            if prepared.zenith_angle is None
            else prepared.zenith_angle[index],
            wavelength=wavelength,
            x=prepared.x,
            y=prepared.y,
        )
        reference = prepared.psfs[index]
        residuals.append(_psf_nrms(reference, prediction.psfs))
        pixel_scale_errors.append(
            abs(
                float(prepared.pixel_scale[index].to_value(u.mas))
                - prediction.pixel_scale.to_value(u.mas)
            )
        )
        plane_metric_errors = _science_metric_errors(
            reference,
            prediction.psfs,
            wavelength=wavelength,
            reference_pixel_scale=float(prepared.pixel_scale[index].to_value(u.mas)),
            prediction_metadata=prediction.metadata,
            tel_diameter=float(prepared.tel_diameter.to_value(u.m)),
            tel_pupil=prepared.tel_pupil,
        )
        for name, values in plane_metric_errors.items():
            metric_errors[name].append(values)
    residual = np.concatenate(residuals)
    metric_error_arrays = {
        name: np.concatenate(values) for name, values in metric_errors.items()
    }
    return ScienceHoPsfReplaySummary(
        psf_nrms_mean=float(np.mean(residual)),
        psf_nrms_max=float(np.max(residual)),
        pixel_scale_abs_max=float(np.max(pixel_scale_errors)) * u.mas,
        metric_rms={
            name: float(np.sqrt(np.mean(values**2)))
            * ({"fwhm": u.mas, "ee": u.dimensionless_unscaled}[name])
            for name, values in metric_error_arrays.items()
        },
        metric_max_abs={
            name: float(np.max(np.abs(values)))
            * ({"fwhm": u.mas, "ee": u.dimensionless_unscaled}[name])
            for name, values in metric_error_arrays.items()
        },
        num_planes=int(prepared.psfs.shape[0]),
        num_points=int(prepared.psfs.shape[1]),
    )


def save_science_ho_psf_interpolator(
    interpolator: ScienceHoPsfInterpolator,
    path: Path,
    *,
    overwrite: bool = False,
) -> None:
    """Save a ready-to-evaluate science-HO-PSF pickle artifact."""
    save_payload(_science_to_payload(interpolator), Path(path), overwrite=overwrite)


def load_science_ho_psf_interpolator(path: Path) -> ScienceHoPsfInterpolator:
    """Load and validate a trusted, ready-to-evaluate science-HO-PSF pickle.

    Pickle reconstruction can execute Python code. Load only artifacts from a
    trusted source.
    """
    return _science_from_payload(load_payload(Path(path)))


# Sample preparation and grid construction


def _prepare_samples(
    samples: ScienceHoPsfSamples, *, dtype: np.dtype
) -> ScienceHoPsfSamples:
    if not isinstance(samples, ScienceHoPsfSamples):
        raise TypeError("samples must be a ScienceHoPsfSamples instance.")
    coordinates = field_coordinates(samples.x, samples.y)
    psfs = _validate_source_psfs(
        samples.psfs, label="psfs", num_points=coordinates.shape[0], dtype=dtype
    )
    num_planes = psfs.shape[0]
    wavelength = _prepare_required_plane_values(
        samples.wavelength,
        unit=u.um,
        label="wavelength",
        length=num_planes,
        positive=True,
        dtype=dtype,
    )
    pixel_scale = _prepare_required_plane_values(
        samples.pixel_scale,
        unit=u.mas,
        label="pixel_scale",
        length=num_planes,
        positive=True,
        dtype=dtype,
    )
    zenith_angle = _prepare_optional_plane_values(
        samples.zenith_angle,
        label="zenith_angle",
        length=num_planes,
        dtype=dtype,
    )
    return ScienceHoPsfSamples(
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
        meta=_prepare_source_meta(samples.meta, num_planes=num_planes, dtype=dtype),
        provenance=tuple(str(value) for value in samples.provenance),
    )


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


def _prepare_required_plane_values(
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
                f"{label} must be scalar or have one value per plane; got length {vector.size}."
            )
    if positive:
        vector = require_positive_vector(
            vector, label=label, length=length, dtype=dtype
        )
    else:
        vector = require_finite_vector(vector, label=label, length=length, dtype=dtype)
    return vector * unit


def _prepare_optional_plane_values(
    value: Any, *, label: str, length: int, dtype=np.float64
) -> u.Quantity | None:
    if value is None:
        return None
    return _prepare_required_plane_values(
        value, unit=u.deg, label=label, length=length, positive=False
    )


def _coordinate_order_for_axes(
    zenith_axis: np.ndarray, wavelength_axis: np.ndarray
) -> tuple[str, ...]:
    order: list[str] = []
    if zenith_axis.size > 1:
        order.append(SCIENCE_COORD_AIRMASS)
    if wavelength_axis.size > 1:
        order.append(SCIENCE_COORD_WAVELENGTH)
    order.extend(SCIENCE_FIELD_COORDINATE_ORDER)
    coordinate_order = tuple(order)
    _validate_coordinate_order(coordinate_order)
    return coordinate_order


def _validate_coordinate_order(coordinate_order: tuple[str, ...]) -> None:
    if tuple(coordinate_order) not in SCIENCE_VALID_COORDINATE_ORDERS:
        raise ValueError(
            f"Unsupported science HO PSF coordinate_order: {tuple(coordinate_order)!r}."
        )


def _active_physical_shape(
    coordinate_order: tuple[str, ...],
    zenith_axis: np.ndarray,
    wavelength_axis: np.ndarray,
) -> tuple[int, ...]:
    shape: list[int] = []
    if SCIENCE_COORD_AIRMASS in coordinate_order:
        shape.append(int(zenith_axis.size))
    if SCIENCE_COORD_WAVELENGTH in coordinate_order:
        shape.append(int(wavelength_axis.size))
    return tuple(shape)


def _active_grid_index(
    coordinate_order: tuple[str, ...],
    zenith_axis: np.ndarray,
    wavelength_axis: np.ndarray,
    zenith_angle: float | None,
    wavelength: u.Quantity,
) -> tuple[int, ...]:
    index: list[int] = []
    if SCIENCE_COORD_AIRMASS in coordinate_order:
        if zenith_angle is None:
            raise ValueError(
                "zenith_angle is required for active airmass interpolation."
            )
        index.append(axis_index(zenith_axis, float(zenith_angle), label="zenith_angle"))
    if SCIENCE_COORD_WAVELENGTH in coordinate_order:
        index.append(axis_index(wavelength_axis, float(wavelength), label="wavelength"))
    return tuple(index)


# Query helpers


def _validate_and_prepare_query_physical_values(
    interpolator: ScienceHoPsfInterpolator,
    *,
    zenith_angle: u.Quantity | None,
    wavelength: u.Quantity | None,
) -> dict[str, float]:
    values: dict[str, float] = {}
    if SCIENCE_COORD_AIRMASS in interpolator.coordinate_order:
        if zenith_angle is None:
            raise ValueError(
                "zenith_angle is required by this science HO PSF artifact."
            )
        values[SCIENCE_COORD_AIRMASS] = float(
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
            dtype=_query_precision(interpolator),
        )

    if SCIENCE_COORD_WAVELENGTH in interpolator.coordinate_order:
        if wavelength is None:
            raise ValueError("wavelength is required by this science HO PSF artifact.")
        values[SCIENCE_COORD_WAVELENGTH] = require_positive_scalar(
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
                dtype=_query_precision(interpolator),
            )
        values[SCIENCE_COORD_WAVELENGTH] = fixed_wavelength
    # A prepared model retains producer rounding at physical support endpoints.
    for name, axis in (
        (SCIENCE_COORD_AIRMASS, interpolator.airmass_axis),
        (SCIENCE_COORD_WAVELENGTH, interpolator.wavelength_axis.to_value(u.um)),
    ):
        if name not in values or name not in interpolator.coordinate_order:
            continue
        bound = (
            2
            * np.finfo(_query_precision(interpolator)).eps
            * max(1.0, abs(float(axis[0])), abs(float(axis[-1])))
        )
        if values[name] < axis[0] - bound or values[name] > axis[-1] + bound:
            raise ValueError(f"{name} outside calibrated support")
        values[name] = interpolator.dtype.type(np.clip(values[name], axis[0], axis[-1]))
    return values


def _query_precision(interpolator):
    """Physical support retains the rounding of the source and runtime state."""
    return support_dtype(
        interpolator.dtype,
        interpolator.builder.get(
            "support_precision", interpolator.builder.get("precision")
        ),
    )


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


def _query_points(
    interpolator: ScienceHoPsfInterpolator,
    physical_values: Mapping[str, float],
    coordinates: np.ndarray,
) -> np.ndarray:
    columns: list[np.ndarray] = []
    for name in interpolator.coordinate_order:
        if name == SCIENCE_COORD_AIRMASS:
            columns.append(
                np.full(
                    coordinates.shape[0],
                    float(physical_values[SCIENCE_COORD_AIRMASS]),
                    dtype=float,
                )
            )
        elif name == SCIENCE_COORD_WAVELENGTH:
            columns.append(
                np.full(
                    coordinates.shape[0],
                    float(physical_values[SCIENCE_COORD_WAVELENGTH]),
                    dtype=float,
                )
            )
        elif name == SCIENCE_COORD_Y:
            columns.append(coordinates[:, 1])
        elif name == SCIENCE_COORD_X:
            columns.append(coordinates[:, 0])
        else:
            raise ValueError(f"Unsupported science HO PSF coordinate {name!r}.")
    return np.column_stack(columns).astype(interpolator.dtype)


def _plane_weights(
    interpolator: ScienceHoPsfInterpolator,
    physical_values: Mapping[str, float],
) -> list[tuple[float, tuple[int, ...]]]:
    weights: list[tuple[float, tuple[int, ...]]] = [(1.0, ())]
    if SCIENCE_COORD_AIRMASS in interpolator.coordinate_order:
        weights = _extend_weights(
            weights,
            interpolation_axis_weights(
                interpolator.airmass_axis,
                physical_values[SCIENCE_COORD_AIRMASS],
                label="airmass",
            ),
        )
    if SCIENCE_COORD_WAVELENGTH in interpolator.coordinate_order:
        weights = _extend_weights(
            weights,
            interpolation_axis_weights(
                interpolator.wavelength_axis,
                physical_values[SCIENCE_COORD_WAVELENGTH],
                label="wavelength",
            ),
        )
    return weights


def _extend_weights(
    current: list[tuple[float, tuple[int, ...]]],
    axis_weights: list[tuple[int, float]],
) -> list[tuple[float, tuple[int, ...]]]:
    return [
        (float(weight * axis_weight), (*index, int(axis_index_value)))
        for weight, index in current
        for axis_index_value, axis_weight in axis_weights
    ]


def _prediction_wavelength(
    interpolator: ScienceHoPsfInterpolator, physical_values: Mapping[str, float]
) -> float:
    if SCIENCE_COORD_WAVELENGTH in interpolator.coordinate_order:
        return float(physical_values[SCIENCE_COORD_WAVELENGTH])
    return float(interpolator.wavelength_axis[0].to_value(u.um))


def _interpolation_axes(
    interpolator: ScienceHoPsfInterpolator,
) -> tuple[np.ndarray, ...]:
    axes: list[np.ndarray] = []
    for name in interpolator.coordinate_order:
        if name == SCIENCE_COORD_AIRMASS:
            axes.append(interpolator.airmass_axis.to_value(u.dimensionless_unscaled))
        elif name == SCIENCE_COORD_WAVELENGTH:
            axes.append(interpolator.wavelength_axis.to_value(u.um))
        elif name == SCIENCE_COORD_Y:
            axes.append(interpolator.y.to_value(u.arcsec))
        elif name == SCIENCE_COORD_X:
            axes.append(interpolator.x.to_value(u.arcsec))
        else:
            raise ValueError(f"Unsupported science HO PSF coordinate {name!r}.")
    return tuple(axes)


def _active_grid_shape(interpolator: ScienceHoPsfInterpolator) -> tuple[int, ...]:
    return _active_physical_shape(
        interpolator.coordinate_order,
        interpolator.zenith_angle_axis.to_value(u.deg),
        interpolator.wavelength_axis.to_value(u.um),
    )


# Validation and replay helpers


def _validate_field_query(
    interpolator: ScienceHoPsfInterpolator, coordinates: np.ndarray
) -> None:
    validate_rectangular_field_query(
        interpolator.x.to_value(u.arcsec),
        interpolator.y.to_value(u.arcsec),
        coordinates,
        label="science HO PSF",
        atol=SCIENCE_HO_PSF_FIELD_QUERY_ATOL.to_value(u.arcsec),
    )


def _snap_field_query(
    interpolator: ScienceHoPsfInterpolator, coordinates: np.ndarray
) -> np.ndarray:
    return snap_rectangular_field_query(
        interpolator.x.to_value(u.arcsec),
        interpolator.y.to_value(u.arcsec),
        coordinates,
    )


def _psf_nrms(reference: np.ndarray, measured: np.ndarray) -> np.ndarray:
    numerator = np.sqrt(
        np.mean((np.asarray(measured) - np.asarray(reference)) ** 2, axis=(-2, -1))
    )
    denominator = np.sqrt(np.mean(np.asarray(reference) ** 2, axis=(-2, -1)))
    return numerator / denominator


def _validate_psf_flux(psfs: np.ndarray, *, label: str) -> None:
    flux = np.sum(np.asarray(psfs, dtype=np.float64), axis=(-2, -1))
    if not np.all(np.isfinite(flux)):
        raise ValueError(f"{label} must have finite per-PSF total flux.")
    if np.any(flux <= 0.0):
        raise ValueError(f"{label} must have strictly positive per-PSF total flux.")


def _prepare_source_meta(
    meta: Mapping[str, Any], *, num_planes: int, dtype=np.float64
) -> dict[str, u.Quantity]:
    if not isinstance(meta, Mapping):
        raise TypeError(
            "meta must be a mapping from field name to scalar or per-plane numeric values."
        )
    prepared: dict[str, u.Quantity] = {}
    for raw_name, raw_value in meta.items():
        name = validate_meta_field_name(raw_name, label="ScienceHoPsfSamples.meta")
        if name in prepared:
            raise ValueError(
                f"ScienceHoPsfSamples.meta contains duplicate field name {name!r}."
            )
        if not isinstance(raw_value, u.Quantity):
            raise TypeError(
                f"ScienceHoPsfSamples.meta[{name!r}] must be an Astropy Quantity."
            )
        value = np.asarray(raw_value.value, dtype=dtype)
        if value.ndim == 0:
            scalar = float(value)
            if not np.isfinite(scalar):
                raise ValueError(f"ScienceHoPsfSamples.meta[{name!r}] must be finite.")
            prepared[name] = scalar * raw_value.unit
            continue
        vector = value.reshape(-1)
        if vector.size != int(num_planes):
            raise ValueError(
                f"ScienceHoPsfSamples.meta[{name!r}] must be scalar or have one value per plane; "
                f"got length {vector.size}, expected {int(num_planes)}."
            )
        if not np.all(np.isfinite(vector)):
            raise ValueError(
                f"ScienceHoPsfSamples.meta[{name!r}] must contain only finite values."
            )
        prepared[name] = vector.astype(dtype, copy=False) * raw_value.unit
    return prepared


def _build_artifact_meta(
    source_meta: Mapping[str, u.Quantity],
    samples: ScienceHoPsfSamples,
    coordinate_order: tuple[str, ...],
    zenith_axis: np.ndarray,
    wavelength_axis: np.ndarray,
    active_shape: tuple[int, ...],
    dtype=np.float64,
) -> dict[str, u.Quantity]:
    artifact_meta: dict[str, u.Quantity] = {}
    for name, value in source_meta.items():
        array = np.asarray(value.value, dtype=dtype)
        if array.ndim == 0:
            artifact_meta[name] = float(array) * value.unit
            continue
        grid = np.full(active_shape, np.nan, dtype=dtype)
        for plane_index, meta_value in enumerate(array.reshape(-1)):
            grid_index = _active_grid_index(
                coordinate_order,
                zenith_axis,
                wavelength_axis,
                None
                if samples.zenith_angle is None
                else float(samples.zenith_angle[plane_index].to_value(u.deg)),
                float(samples.wavelength[plane_index].to_value(u.um)),
            )
            grid[grid_index] = float(meta_value)
        artifact_meta[name] = grid * value.unit
    return artifact_meta


def _initial_evaluated_meta(
    meta: Mapping[str, Any], *, grid_shape: tuple[int, ...], dtype=np.float64
) -> dict[str, u.Quantity]:
    evaluated: dict[str, u.Quantity] = {}
    for name, value in _validate_artifact_meta(meta, grid_shape=grid_shape).items():
        array = np.asarray(value.value, dtype=dtype)
        if array.ndim == 0:
            evaluated[name] = float(array) * value.unit
        else:
            evaluated[name] = 0.0 * value.unit
    return evaluated


def _accumulate_grid_meta(
    evaluated: dict[str, u.Quantity],
    meta: Mapping[str, Any],
    *,
    weight: float,
    grid_index: tuple[int, ...],
) -> None:
    for name, value in meta.items():
        array = np.asarray(value.value, dtype=float)
        if array.ndim == 0:
            continue
        evaluated[name] += float(weight) * float(array[grid_index]) * value.unit


def _validate_artifact_meta(
    meta: Mapping[str, Any], *, grid_shape: tuple[int, ...] | None
) -> dict[str, u.Quantity]:
    if not isinstance(meta, Mapping):
        raise TypeError("metadata.meta must be a mapping.")
    validated: dict[str, u.Quantity] = {}
    for raw_name, raw_value in meta.items():
        name = validate_meta_field_name(raw_name, label="metadata.meta")
        if name in validated:
            raise ValueError(f"metadata.meta contains duplicate field name {name!r}.")
        if not isinstance(raw_value, u.Quantity):
            raise TypeError(f"metadata.meta[{name!r}] must be an Astropy Quantity.")
        value = np.asarray(raw_value.value, dtype=float)
        if value.ndim == 0:
            scalar = float(value)
            if not np.isfinite(scalar):
                raise ValueError(f"metadata.meta[{name!r}] must be finite.")
            validated[name] = scalar * raw_value.unit
            continue
        if grid_shape is not None and value.shape != grid_shape:
            raise ValueError(
                f"metadata.meta[{name!r}] must be scalar or have shape {grid_shape}; got {value.shape}."
            )
        if not np.all(np.isfinite(value)):
            raise ValueError(
                f"metadata.meta[{name!r}] must contain only finite values."
            )
        validated[name] = np.asarray(value, dtype=float) * raw_value.unit
    return validated


def _science_metric_errors(
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
        raise ValueError("Science HO PSF replay could not compute finite fwhm metrics.")

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
        raise ValueError("Science HO PSF replay could not compute finite ee metrics.")
    return {
        "fwhm": measured_fwhm - reference_fwhm,
        "ee": measured_ee - reference_ee,
    }


# Persistence and validation


def _interpolation_method(coordinate_order: tuple[str, ...]) -> str:
    label_by_coordinate = {
        SCIENCE_COORD_AIRMASS: "airmass",
        SCIENCE_COORD_WAVELENGTH: "wavelength",
        SCIENCE_COORD_Y: "y",
        SCIENCE_COORD_X: "x",
    }
    return "regular_grid_linear_" + "_".join(
        label_by_coordinate[name] for name in coordinate_order
    )


def _science_to_payload(interpolator: ScienceHoPsfInterpolator) -> dict[str, Any]:
    if not isinstance(interpolator, ScienceHoPsfInterpolator):
        raise TypeError("interpolator must be a ScienceHoPsfInterpolator instance.")
    _validate_interpolator(interpolator)
    prepared = copy(interpolator._prepared_runtime().psf_interpolator)
    # SciPy has moved this helper between private modules; NumPy input is the
    # supported query contract and keeps the pickle usable across that move.
    prepared._asarray = np.asarray
    return {
        "kind": SCIENCE_HO_PSF_ARTIFACT_KIND,
        "version": SCIENCE_HO_PSF_ARTIFACT_VERSION,
        "builder": dict(interpolator.builder),
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
            "meta": dict(interpolator.meta),
            "provenance": tuple(interpolator.provenance),
        },
        "model": {"prepared_psf_interpolator": prepared},
    }


def _science_from_payload(payload: Mapping[str, Any]) -> ScienceHoPsfInterpolator:
    validate_payload_kind(
        payload,
        kind=SCIENCE_HO_PSF_ARTIFACT_KIND,
        version=SCIENCE_HO_PSF_ARTIFACT_VERSION,
    )
    interpolation = dict(payload.get("interpolation", {}))
    coordinate_order = tuple(
        str(value) for value in interpolation.get("coordinate_order", ())
    )
    _validate_coordinate_order(coordinate_order)
    if interpolation.get("method") != _interpolation_method(coordinate_order):
        raise ValueError("Unsupported science-HO-PSF interpolation method.")
    model = dict(payload.get("model", {}))
    prepared = model.get("prepared_psf_interpolator")
    if not isinstance(prepared, RegularGridInterpolator):
        raise TypeError(
            "Science-HO-PSF artifact requires a prepared RegularGridInterpolator."
        )
    psf_grid = np.asarray(prepared.values)
    dtype = validate_psf_dtype(psf_grid.dtype)
    metadata = dict(payload.get("metadata", {}))
    interpolator = ScienceHoPsfInterpolator(
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
        meta=dict(metadata.get("meta", {})),
        psf_grid=psf_grid,
        dtype=dtype,
        provenance=tuple(str(value) for value in metadata.get("provenance", ())),
        builder=dict(payload.get("builder", {})),
    )
    _validate_interpolator(interpolator)
    _validate_prepared_interpolator(interpolator, prepared)
    psf_grid.flags.writeable = False
    object.__setattr__(interpolator, "psf_grid", psf_grid)
    object.__setattr__(interpolator, "_prepared_psf_interpolator", prepared)
    return interpolator


def _validate_prepared_interpolator(
    interpolator: ScienceHoPsfInterpolator,
    prepared: RegularGridInterpolator,
) -> None:
    if prepared.method != "linear" or prepared.bounds_error is not True:
        raise ValueError(
            "Science-HO-PSF prepared interpolator must use bounded linear interpolation."
        )
    axes = _interpolation_axes(interpolator)
    if len(prepared.grid) != len(axes) or any(
        not np.array_equal(stored, expected)
        for stored, expected in zip(prepared.grid, axes, strict=True)
    ):
        raise ValueError(
            "Science-HO-PSF prepared interpolator axes do not match artifact metadata."
        )


def _validate_interpolator(interpolator: ScienceHoPsfInterpolator) -> None:
    _validate_coordinate_order(tuple(interpolator.coordinate_order))
    if len(interpolator.psf_shape) != 2 or any(
        int(value) <= 0 for value in interpolator.psf_shape
    ):
        raise ValueError("metadata.psf_shape must contain two positive dimensions.")
    if interpolator.airmass_axis.shape != interpolator.zenith_angle_axis.shape:
        raise ValueError("airmass_axis shape must match zenith_angle_axis shape.")
    if SCIENCE_COORD_AIRMASS in interpolator.coordinate_order:
        if interpolator.zenith_angle_axis.size <= 1:
            raise ValueError(
                "active airmass coordinate requires multiple zenith_angle_axis values."
            )
    elif interpolator.zenith_angle_axis.size > 1:
        raise ValueError(
            "multiple zenith_angle_axis values require active airmass coordinate."
        )
    if SCIENCE_COORD_WAVELENGTH in interpolator.coordinate_order:
        if interpolator.wavelength_axis.size <= 1:
            raise ValueError(
                "active wavelength coordinate requires multiple wavelength_axis values."
            )
    elif interpolator.wavelength_axis.size != 1:
        raise ValueError(
            "fixed wavelength artifacts must store exactly one wavelength_axis value."
        )
    if interpolator.x.ndim != 1 or interpolator.y.ndim != 1:
        raise ValueError("x and y must be 1-D rectangular field axes.")
    if interpolator.x.size == 0 or interpolator.y.size == 0:
        raise ValueError("x and y field axes must not be empty.")
    if np.any(np.diff(interpolator.x) <= 0.0) or np.any(np.diff(interpolator.y) <= 0.0):
        raise ValueError("x and y field axes must be strictly increasing.")
    active_shape = _active_grid_shape(interpolator)
    pixel_scale_grid = quantity_value(
        interpolator.pixel_scale_grid, u.mas, label="pixel_scale_grid", dtype=float
    )
    if pixel_scale_grid.shape != active_shape:
        raise ValueError(f"pixel_scale_grid shape must be {active_shape}.")
    if not np.all(np.isfinite(pixel_scale_grid)):
        raise ValueError("pixel_scale_grid must contain only finite values.")
    _validate_artifact_meta(interpolator.meta, grid_shape=active_shape)
    expected_psf_grid_shape = _expected_model_grid_shape(interpolator)
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


def _expected_model_grid_shape(
    interpolator: ScienceHoPsfInterpolator,
) -> tuple[int, ...]:
    sizes = {
        SCIENCE_COORD_AIRMASS: interpolator.airmass_axis.size,
        SCIENCE_COORD_WAVELENGTH: interpolator.wavelength_axis.size,
        SCIENCE_COORD_Y: interpolator.y.size,
        SCIENCE_COORD_X: interpolator.x.size,
    }
    return tuple(int(sizes[name]) for name in interpolator.coordinate_order) + tuple(
        interpolator.psf_shape
    )
