"""Measured gain/variance interpolation and reusable paired noise banks."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np
from astropy import units as u
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


def _immutable_array(value, dtype=None):
    value = np.ascontiguousarray(value, dtype=dtype)
    return np.frombuffer(value.tobytes(), dtype=value.dtype).reshape(value.shape)


@dataclass(frozen=True, kw_only=True, eq=False)
class SensingPrediction(_PhysicalRecord):
    """Per-NGS local response and conditional noise for supplied physical inputs.

    Attributes:
        gain: Positive dimensionless scalar gains, shape (NGS,).
        variance: Positive per-axis angular noise variances, matching gain.
    """

    gain: u.Quantity
    variance: u.Quantity

    def __post_init__(self):
        gain = _quantity(self.gain, u.one, "gain", 1)
        variance = _quantity(self.variance, u.mas**2, "variance", 1)
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

    def __init__(self):
        raise TypeError("Use build_sensing_response or load_sensing_calibration")

    @classmethod
    def _from_arrays(cls, arrays, controller, photometry, field_radius, intervals):
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
                axes[0], 1 / np.cos(np.deg2rad(angles)), rtol=1e-12, atol=0
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
        radius = _quantity(field_radius, u.arcsec, "field_radius", 0)
        if radius.value <= 0:
            raise ValueError("field_radius must be positive")
        intervals = np.array(calibration_magnitude_intervals(axes[3], intervals), float)
        saved = MappingProxyType(
            {key: _immutable_array(value) for key, value in arrays.items()}
        )
        result = object.__new__(cls)
        for name, value in (
            ("_arrays", saved),
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
                    saved[key].astype(float) if key == "support" else saved[key],
                    bounds_error=True,
                )
                for key in ("gain", "log_variance_mas2", "support")
            ),
        )
        return result

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
        positions = _quantity(positions, u.arcsec, "positions", 2)
        angle = _angle(zenith_angle)
        if positions.shape[1] != 2:
            raise ValueError("positions must have shape (N,2)")
        if not isinstance(magnitudes, u.Quantity):
            raise TypeError("magnitudes must be a physical quantity")
        mag = np.asarray(magnitudes.to_value(u.mag))
        if (
            mag.dtype.kind not in "fiu"
            or not np.isfinite(mag).all()
            or mag.shape not in ((), (len(positions),))
        ):
            raise ValueError("finite scalar or per-NGS magnitudes required")
        mag = np.broadcast_to(mag, (len(positions),))
        if np.any(
            np.linalg.norm(positions.value, axis=1) > self.field_radius.value + 1e-8
        ):
            raise ValueError("position outside calibrated sensing field support")
        allowed = np.zeros(len(positions), bool)
        for low, high in self._intervals:
            allowed |= (mag >= low - 1e-10) & (mag <= high + 1e-10)
        if not allowed.all():
            raise ValueError(
                "magnitude outside measured sensing intervals or in a calibration gap"
            )
        angles = self.zenith_angles.value
        if angle.value < angles[0] or angle.value > angles[-1]:
            raise ValueError("zenith angle outside calibrated sensing support")
        x, y = positions.value.T
        for name, value, axis in (
            ("x", x, self.x.value),
            ("y", y, self.y.value),
            ("magnitude", mag, self.magnitudes.value),
        ):
            if np.any(value < axis[0]) or np.any(value > axis[-1]):
                raise ValueError(f"query outside calibrated sensing {name} support")
        query = np.column_stack(
            (
                np.full(len(positions), 1 / np.cos(np.deg2rad(angle.value))),
                positions.value,
                mag,
            )
        )
        gain, log_variance, support = self._interpolators
        if np.any(support(query) < 1 - 1e-12):
            raise ValueError("requested sensing cell requires unmeasured vertices")
        return SensingPrediction(
            gain=gain(query) * u.one, variance=np.exp(log_variance(query)) * u.mas**2
        )


@dataclass(frozen=True, init=False, eq=False)
class SensingResponseCalibration:
    """Prepared response model and paired dimensionless banks for offline replay.

    Build from reusable WFS observations with build_sensing_response. The banks
    preserve the accepted conditional whitening and float32 sample precision;
    they are not included in the runtime sensing model. draw_noise selects a
    rectangular mixture at an actually measured zenith plane.
    """

    _model: SensingResponseModel
    _banks: Mapping
    _sampling: object
    _displacement: u.Quantity

    def __init__(self):
        raise TypeError("Use build_sensing_response")

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
        position = _quantity(position, u.arcsec, "position", 1)
        if position.shape != (2,):
            raise ValueError("position must contain x,y")
        mag = _quantity(magnitude, u.mag, "magnitude", 0)
        angle = _angle(zenith_angle)
        _count(count, "count")
        if not isinstance(rng, np.random.Generator):
            raise TypeError("rng must be a NumPy Generator")
        self.model.predict(
            positions=position[None, :], magnitudes=mag, zenith_angle=angle
        )
        if angle.value not in self.model.zenith_angles.value:
            raise ValueError(
                "noise-bank draw requires an actually measured zenith plane"
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
            len(neighbors), count, p=[weight for _, weight in neighbors]
        )
        result = np.empty((count, 2))
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
) -> SensingResponseCalibration:
    """Build scalar gain/log-variance tables and replay banks without native access.

    Stream captured points, requiring unique coordinates and compatible numerical
    sampling, photometry and controller settings. Preserve gaps unless explicit
    intervals declare calibrated nonuniform magnitude spacing. The runtime
    sensing product is not publishable until its noise-shape fit is complete.
    """
    rows = []
    banks = {}
    first = None
    for item in measurements:
        if not isinstance(item, SensingCalibrationMeasurements):
            raise TypeError("typed sensing observations required")
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
        banks[point] = _immutable_array(white, np.float32)
        rows.append((*point, gain, np.log(variance)))
    if not rows:
        raise ValueError("no sensing calibration observations")
    values = np.asarray(rows, float)
    angles = np.unique(values[:, 0])
    x = np.unique(values[:, 1])
    y = np.unique(values[:, 2])
    mag = np.unique(values[:, 3])
    shape = (len(angles), len(x), len(y), len(mag))
    gain = np.zeros(shape)
    variance = np.zeros(shape)
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
    )
    result = object.__new__(SensingResponseCalibration)
    object.__setattr__(result, "_model", model)
    object.__setattr__(result, "_banks", MappingProxyType(banks))
    object.__setattr__(result, "_sampling", first.sampling)
    object.__setattr__(result, "_displacement", first.displacement)
    return result
