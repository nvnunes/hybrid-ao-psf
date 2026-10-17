"""Measured gain/variance interpolation and reusable paired noise banks."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, replace
from types import MappingProxyType

import numpy as np
from astropy import units as u
from numpy.typing import DTypeLike
from scipy.interpolate import RegularGridInterpolator

from ._calibration_contracts import (
    CalibrationPhotometry,
    FixedGainIntegrator,
    _angle,
    _count,
    _PhysicalRecord,
    _quantity,
)
from ._sensing_grid import calibration_magnitude_intervals, rectangular_neighbors
from ._sensing_measurements import SensingCalibrationMeasurements
from ._validation import clip_support, support_dtype, validate_psf_dtype


def _immutable_array(value, dtype=None):
    value = np.ascontiguousarray(value, dtype=dtype)
    return np.frombuffer(value.tobytes(), dtype=value.dtype).reshape(value.shape)


@dataclass(frozen=True, kw_only=True, eq=False)
class SensingPrediction(_PhysicalRecord):
    """Per-NGS local response and conditional noise for supplied physical inputs.

    Attributes:
        dtype: Real calculation/state precision, float32 by default or float64.
        gain: Positive dimensionless scalar gains, shape (NGS,).
        variance: Positive per-axis angular noise variances, matching gain.
    """

    gain: u.Quantity
    variance: u.Quantity

    dtype: DTypeLike = field(default=np.float32, kw_only=True)

    def __post_init__(self):
        object.__setattr__(self, "dtype", validate_psf_dtype(self.dtype))
        gain = _quantity(self.gain, u.one, "gain", 1, dtype=self.dtype)
        variance = _quantity(self.variance, u.mas**2, "variance", 1, dtype=self.dtype)
        if (
            gain.shape != variance.shape
            or np.any(gain.value <= 0)
            or np.any(variance.value <= 0)
        ):
            raise ValueError("aligned positive sensing gain/variance required")
        object.__setattr__(self, "gain", gain)
        object.__setattr__(self, "variance", variance)


@dataclass(frozen=True, init=False, eq=False)
class SensingResponseModel:
    """Immutable gain/log-variance tables over measured position/magnitude/airmass.

    Built by build_sensing_response; complete runtime persistence belongs to
    SensingCalibration after the noise-shape fit. Queries use physical positions,
    magnitudes and viewing angle. Missing vertices, magnitude gaps and spatial
    or airmass extrapolation are rejected. No corner truth is invented.
    """

    _arrays: Mapping
    _interpolators: tuple
    _controller: FixedGainIntegrator
    _photometry: CalibrationPhotometry
    _field_radius: u.Quantity
    _intervals: np.ndarray
    _producer_dtype: np.dtype | None
    _support_precision: np.dtype

    def __init__(self):
        raise TypeError("Use build_sensing_response or load_sensing_calibration")

    @classmethod
    def _from_arrays(
        cls,
        arrays,
        controller,
        photometry,
        field_radius,
        intervals,
        producer_dtype=None,
        support_precision=None,
    ):
        dtype = validate_psf_dtype(arrays["gain"].dtype)
        support_precision = support_dtype(dtype, producer_dtype, support_precision)
        expected = {
            "airmass",
            "zenith_deg",
            "x_arcsec",
            "y_arcsec",
            "magnitude",
            "gain",
            "log_variance_mas2",
            "support",
        }
        if set(arrays) != expected:
            raise ValueError("unexpected sensing interpolation arrays")
        axes = tuple(
            np.asarray(arrays[key])
            for key in ("airmass", "x_arcsec", "y_arcsec", "magnitude")
        )
        if any(
            a.ndim != 1
            or not len(a)
            or a.dtype.kind not in "fiu"
            or not np.isfinite(a).all()
            or np.any(np.diff(a) <= 0)
            for a in axes
        ):
            raise ValueError("finite increasing sensing interpolation axes required")
        if any(len(a) < 2 for a in axes[1:]):
            raise ValueError("at least two spatial/magnitude nodes required")
        angles = np.asarray(arrays["zenith_deg"])
        if (
            angles.shape != axes[0].shape
            or not np.isfinite(angles).all()
            or np.any(angles < 0)
            or np.any(angles >= 90)
            or np.any(np.diff(angles) <= 0)
            or not np.allclose(
                axes[0],
                1 / np.cos(np.deg2rad(angles)),
                rtol=max(1e-12, 4 * np.finfo(support_precision).eps),
                atol=0,
            )
        ):
            raise ValueError("measured sensing zenith/airmass axes disagree")
        shape = tuple(len(a) for a in axes)
        gain, log_variance, support = (
            np.asarray(arrays[key]) for key in ("gain", "log_variance_mas2", "support")
        )
        if (
            any(v.shape != shape for v in (gain, log_variance, support))
            or not np.isfinite(gain).all()
            or not np.isfinite(log_variance).all()
            or support.dtype != bool
            or not np.any(support)
            or np.any(gain[support] <= 0)
        ):
            raise ValueError("invalid supported sensing table values")
        if not isinstance(controller, FixedGainIntegrator) or not isinstance(
            photometry, CalibrationPhotometry
        ):
            raise TypeError("typed controller and photometry required")
        radius = _quantity(field_radius, u.arcsec, "field_radius", 0, dtype=dtype)
        if radius.value <= 0:
            raise ValueError("field_radius must be positive")
        intervals = np.array(
            calibration_magnitude_intervals(axes[3], intervals), dtype=dtype
        )
        saved = MappingProxyType(
            {key: _immutable_array(value) for key, value in arrays.items()}
        )
        result = object.__new__(cls)
        for name, value in (
            ("_arrays", saved),
            ("_support_precision", support_precision),
            (
                "_producer_dtype",
                None if producer_dtype is None else validate_psf_dtype(producer_dtype),
            ),
            ("_controller", controller),
            ("_photometry", photometry),
            ("_field_radius", radius),
            ("_intervals", _immutable_array(intervals)),
        ):
            object.__setattr__(result, name, value)
        frozen_axes = tuple(
            saved[key] for key in ("airmass", "x_arcsec", "y_arcsec", "magnitude")
        )
        object.__setattr__(
            result,
            "_interpolators",
            tuple(
                RegularGridInterpolator(
                    frozen_axes,
                    saved[key].astype(dtype) if key == "support" else saved[key],
                    bounds_error=True,
                )
                for key in ("gain", "log_variance_mas2", "support")
            ),
        )
        return result

    @property
    def dtype(self) -> np.dtype:
        """Precision of the loaded or prepared sensor table."""
        return self._arrays["gain"].dtype

    @property
    def producer_dtype(self) -> np.dtype | None:
        """Recorded calibration precision; None for unlabelled older products."""
        return self._producer_dtype

    @property
    def support_precision(self) -> np.dtype:
        """Coarsest coordinate precision retained through preparation and loading."""
        return self._support_precision

    def prepare(self, *, dtype=np.float32) -> SensingResponseModel:
        """Copy runtime state and preserve the original producer precision."""
        dtype = validate_psf_dtype(dtype)
        arrays = {
            key: value.astype(dtype) if value.dtype.kind == "f" else value.copy()
            for key, value in self._arrays.items()
        }
        return self._from_arrays(
            arrays,
            replace(self.controller, dtype=dtype),
            replace(self.photometry, dtype=dtype),
            self.field_radius,
            self._intervals.astype(dtype),
            self.producer_dtype,
            self.support_precision,
        )

    @property
    def controller(self) -> FixedGainIntegrator:
        """Return the fixed numerical controller on which the shape calibration rests."""
        return self._controller

    @property
    def photometry(self) -> CalibrationPhotometry:
        """Return the measured magnitude convention and incident-count conversion."""
        return self._photometry

    @property
    def field_radius(self) -> u.Quantity:
        """Return the declared calibrated circular field radius."""
        return self._field_radius.view()

    @property
    def magnitude_intervals(self) -> u.Quantity:
        """Return disjoint measured support intervals, including any deliberate gaps."""
        return u.Quantity(self._intervals, u.mag, copy=False)

    @property
    def zenith_angles(self) -> u.Quantity:
        """Return the actual measured viewing-angle nodes."""
        return u.Quantity(self._arrays["zenith_deg"], u.deg, copy=False)

    @property
    def x(self) -> u.Quantity:
        """Return rectangular Cartesian x nodes in arcseconds."""
        return u.Quantity(self._arrays["x_arcsec"], u.arcsec, copy=False)

    @property
    def y(self) -> u.Quantity:
        """Return rectangular Cartesian y nodes in arcseconds."""
        return u.Quantity(self._arrays["y_arcsec"], u.arcsec, copy=False)

    @property
    def magnitudes(self) -> u.Quantity:
        """Return the measured magnitude nodes in the photometric convention."""
        return u.Quantity(self._arrays["magnitude"], u.mag, copy=False)

    @property
    def gain_table(self) -> u.Quantity:
        """Return dimensionless gain on (airmass,x,y,magnitude), masked elsewhere."""
        return u.Quantity(self._arrays["gain"], u.one, copy=False)

    @property
    def variance_table(self) -> u.Quantity:
        """Return per-axis variance in mas squared on the rectangular support grid.

        Values outside support_mask are numerical scaffolding, not measurements.
        """
        return np.exp(self._arrays["log_variance_mas2"]) * u.mas**2

    @property
    def support_mask(self) -> np.ndarray:
        """Return immutable measured-vertex membership on the rectangular grid."""
        return self._arrays["support"].view()

    def predict(
        self, *, positions: u.Quantity, magnitudes: u.Quantity, zenith_angle: u.Quantity
    ) -> SensingPrediction:
        """Interpolate local scalar gain and per-axis variance for one NGS list.

        positions is (N,2); magnitudes is length N or a scalar shared by all NGS.
        Viewing angle is scalar. Enforce measured intervals, field radius,
        rectangular bounds and support for every nonzero-weight vertex.
        """
        precision = self.support_precision
        positions = _quantity(positions, u.arcsec, "positions", 2, dtype=self.dtype)
        angle = _angle(zenith_angle, dtype=self.dtype)
        if positions.shape[1] != 2:
            raise ValueError("positions must have shape (N,2)")
        if not isinstance(magnitudes, u.Quantity):
            raise TypeError("magnitudes must be a physical quantity")
        mag = np.asarray(magnitudes.to_value(u.mag), dtype=self.dtype)
        if (
            mag.dtype.kind not in "fiu"
            or not np.isfinite(mag).all()
            or mag.shape not in ((), (len(positions),))
        ):
            raise ValueError("finite scalar or per-NGS magnitudes required")
        mag = np.broadcast_to(mag, (len(positions),))
        if np.any(
            np.linalg.norm(positions.value, axis=1)
            > self.field_radius.value
            + max(1e-8, 4 * np.finfo(precision).eps * self.field_radius.value)
        ):
            raise ValueError("position outside calibrated sensing field support")
        allowed = np.zeros(len(positions), bool)
        for low, high in self._intervals:
            bound = max(
                1e-10, 2 * np.finfo(precision).eps * max(1.0, abs(low), abs(high))
            )
            allowed |= (mag >= low - bound) & (mag <= high + bound)
        if not allowed.all():
            raise ValueError(
                "magnitude outside measured sensing intervals or in a calibration gap"
            )
        angles = self.zenith_angles.value
        angle = (
            clip_support(angle.value, angles, dtype=precision, label="zenith angle")
            * u.deg
        )
        x, y = positions.value.T
        x = clip_support(x, self.x.value, dtype=precision, label="sensing x")
        y = clip_support(y, self.y.value, dtype=precision, label="sensing y")
        mag = clip_support(
            mag, self.magnitudes.value, dtype=precision, label="sensing magnitude"
        )
        query = np.column_stack(
            (
                np.full(
                    len(positions),
                    np.clip(
                        1 / np.cos(np.deg2rad(angle.value)),
                        self._arrays["airmass"][0],
                        self._arrays["airmass"][-1],
                    ),
                    dtype=self.dtype,
                ),
                x,
                y,
                mag,
            )
        )
        gain, log_variance, support = self._interpolators
        if np.any(support(query) < 1 - 1e-12):
            raise ValueError("requested sensing cell requires unmeasured vertices")
        return SensingPrediction(
            dtype=self.dtype,
            gain=np.asarray(gain(query), dtype=self.dtype) * u.one,
            variance=np.exp(np.asarray(log_variance(query), dtype=self.dtype))
            * u.mas**2,
        )


@dataclass(frozen=True, init=False, eq=False)
class SensingResponseCalibration:
    """Prepared response model and paired dimensionless banks for offline replay.

    Build from reusable WFS observations with build_sensing_response. The banks
    preserve the accepted conditional whitening at their configured precision;
    they are not included in the runtime sensing model. draw_noise selects a
    rectangular mixture at an actually measured zenith plane.
    """

    _model: SensingResponseModel
    _banks: Mapping
    _sampling: object
    _displacement: u.Quantity

    def __init__(self):
        raise TypeError("Use build_sensing_response")

    def prepare(self, *, dtype=np.float32) -> SensingResponseCalibration:
        """Copy model and banks at one precision; reject collapsed bank nodes."""
        dtype = validate_psf_dtype(dtype)
        banks = {}
        for key, value in self._banks.items():
            prepared_key = tuple(map(float, np.asarray(key, dtype=dtype)))
            if prepared_key in banks:
                raise ValueError(
                    "precision conversion collapses measured noise-bank nodes"
                )
            banks[prepared_key] = _immutable_array(value, dtype)
        result = object.__new__(type(self))
        object.__setattr__(result, "_model", self.model.prepare(dtype=dtype))
        object.__setattr__(result, "_banks", MappingProxyType(banks))
        object.__setattr__(result, "_sampling", self._sampling)
        object.__setattr__(
            result,
            "_displacement",
            _quantity(self._displacement, u.mas, "displacement", 0, dtype=dtype),
        )
        return result

    @property
    def model(self) -> SensingResponseModel:
        """Return the immutable gain/variance provider for the captured observations."""
        return self._model

    @property
    def bank_count(self) -> int:
        """Return the number of captured direction/magnitude/zenith noise banks."""
        return len(self._banks)

    def draw_noise(
        self,
        *,
        position: u.Quantity,
        magnitude: u.Quantity,
        zenith_angle: u.Quantity,
        rng: np.random.Generator,
        count: int,
    ) -> np.ndarray:
        """Draw count paired white-noise samples using existing rectangular weights.

        The generator is caller-owned and advanced deterministically. Return
        newly owned (count,2) dimensionless samples. Viewing angle must be a
        captured plane; no between-plane noise-shape interpolation is invented.
        """
        position = _quantity(position, u.arcsec, "position", 1, dtype=self.model.dtype)
        if position.shape != (2,):
            raise ValueError("position must contain x,y")
        mag = _quantity(magnitude, u.mag, "magnitude", 0, dtype=self.model.dtype)
        angle = _angle(zenith_angle, dtype=self.model.dtype)
        _count(count, "count")
        if not isinstance(rng, np.random.Generator):
            raise TypeError("rng must be a NumPy Generator")
        self.model.predict(
            positions=position[None, :], magnitudes=mag, zenith_angle=angle
        )
        precision = self.model.support_precision
        measured_angles = self.model.zenith_angles.value
        bound = 2 * np.finfo(precision).eps * np.maximum(1.0, np.abs(measured_angles))
        matches = np.flatnonzero(np.abs(measured_angles - angle.value) <= bound)
        if len(matches) != 1:
            raise ValueError(
                "noise-bank draw requires an actually measured zenith plane"
            )
        angle = measured_angles[matches[0]] * u.deg
        position = (
            np.array(
                [
                    clip_support(
                        position.value[0],
                        self.model.x.value,
                        dtype=precision,
                        label="sensing x",
                    ),
                    clip_support(
                        position.value[1],
                        self.model.y.value,
                        dtype=precision,
                        label="sensing y",
                    ),
                ]
            )
            * u.arcsec
        )
        mag = (
            clip_support(
                mag.value,
                self.model.magnitudes.value,
                dtype=precision,
                label="sensing magnitude",
            )
            * u.mag
        )
        axes = (self.model.x.value, self.model.y.value, self.model.magnitudes.value)
        neighbors = rectangular_neighbors([*position.value, float(mag.value)], axes)
        if any(
            (float(angle.value), *point) not in self._banks
            for point, weight in neighbors
            if weight > 0
        ):
            raise ValueError(
                "noise-bank interpolation requires every nonzero-weight measured vertex"
            )
        indices = rng.choice(
            len(neighbors),
            count,
            p=np.asarray([weight for _, weight in neighbors], dtype=self.model.dtype),
        )
        result = np.empty((count, 2), dtype=self.model.dtype)
        for index, (point, _weight) in enumerate(neighbors):
            selected = np.flatnonzero(indices == index)
            bank = self._banks[(float(angle.value), *point)]
            result[selected] = bank[rng.integers(len(bank), size=len(selected))]
        return result


def build_sensing_response(
    measurements: Iterable[SensingCalibrationMeasurements],
    *,
    field_radius: u.Quantity,
    magnitude_intervals: u.Quantity | None = None,
    dtype=np.float32,
) -> SensingResponseCalibration:
    """Build scalar gain/log-variance tables and replay banks without native access.

    Stream captured points, requiring unique coordinates and compatible numerical
    sampling, photometry and controller settings. Preserve gaps unless explicit
    intervals declare calibrated nonuniform magnitude spacing. The runtime
    sensing product is not publishable until its noise-shape fit is complete.
    """
    dtype = validate_psf_dtype(dtype)
    rows = []
    banks = {}
    first = None
    coordinate_precision = dtype
    for item in measurements:
        if not isinstance(item, SensingCalibrationMeasurements):
            raise TypeError("typed sensing observations required")
        coordinate_precision = support_dtype(coordinate_precision, item.dtype)
        item = replace(
            item,
            dtype=dtype,
            samples=replace(item.samples, dtype=dtype),
            controller=replace(item.controller, dtype=dtype),
            photometry=replace(item.photometry, dtype=dtype),
        )
        if first is None:
            first = item
        if (
            item.controller.gain != first.controller.gain
            or item.controller.delay != first.controller.delay
            or item.controller.frame_rate != first.controller.frame_rate
            or item.photometry.magnitude_band != first.photometry.magnitude_band
            or item.photometry.photons_per_frame_zeropoint
            != first.photometry.photons_per_frame_zeropoint
            or item.displacement != first.displacement
            or any(
                getattr(item.sampling, key) != getattr(first.sampling, key)
                for key in ("zero_draws", "gain_draws", "block_draws")
            )
        ):
            raise ValueError("incompatible sensing sampling, photometry or controller")
        point = (
            float(item.zenith_angle.value),
            *map(float, item.position.value),
            float(item.magnitude.value),
        )
        if point in banks:
            raise ValueError("duplicate sensing calibration point")
        stats = item.statistics()
        gain = stats.gain
        variance = float(stats.variance.value)
        if not np.isfinite(gain) or gain <= 0 or variance <= 0:
            raise ValueError("positive measured sensing gains/variances required")
        white = stats.white.value.copy()
        white -= white.mean(0)
        white = np.linalg.solve(
            np.linalg.cholesky(white.T @ white / len(white)), white.T
        ).T
        banks[point] = _immutable_array(white, dtype)
        rows.append((*point, gain, np.log(dtype.type(variance))))
    if not rows:
        raise ValueError("no sensing calibration observations")
    values = np.asarray(rows, dtype=dtype)
    angles = np.unique(values[:, 0])
    x = np.unique(values[:, 1])
    y = np.unique(values[:, 2])
    mag = np.unique(values[:, 3])
    shape = (len(angles), len(x), len(y), len(mag))
    gain = np.zeros(shape, dtype=dtype)
    variance = np.zeros(shape, dtype=dtype)
    support = np.zeros(shape, bool)
    for row in rows:
        index = tuple(
            int(np.searchsorted(axis, value))
            for axis, value in zip((angles, x, y, mag), row[:4], strict=True)
        )
        gain[index] = row[4]
        variance[index] = row[5]
        support[index] = True
    declared = (
        None
        if magnitude_intervals is None
        else _quantity(magnitude_intervals, u.mag, "magnitude_intervals", 2).value
    )
    model = SensingResponseModel._from_arrays(
        {
            "airmass": 1 / np.cos(np.deg2rad(angles)),
            "zenith_deg": angles,
            "x_arcsec": x,
            "y_arcsec": y,
            "magnitude": mag,
            "gain": gain,
            "log_variance_mas2": variance,
            "support": support,
        },
        first.controller,
        first.photometry,
        field_radius,
        declared,
        producer_dtype=dtype,
        support_precision=coordinate_precision,
    )
    result = object.__new__(SensingResponseCalibration)
    object.__setattr__(result, "_model", model)
    object.__setattr__(result, "_banks", MappingProxyType(banks))
    object.__setattr__(result, "_sampling", first.sampling)
    object.__setattr__(result, "_displacement", first.displacement)
    return result
