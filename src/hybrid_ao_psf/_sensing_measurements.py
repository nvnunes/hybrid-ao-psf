"""Physical fixed-spot observations and conditional paired-WFS statistics."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from astropy import units as u
from numpy.typing import DTypeLike

from ._calibration_contracts import (
    CalibrationPhotometry,
    FixedGainIntegrator,
    SensingWfsMeasurements,
    SensingWfsSampling,
    _angle,
    _PhysicalRecord,
    _quantity,
)
from ._calibration_io import validate_observation_precision
from ._calibration_products import (
    controller_header,
    photometry_header,
    read_controller_header,
    read_photometry_header,
)
from ._io import atomic_output_path
from ._sensing_statistics import measurement_statistics
from ._validation import validate_psf_dtype

_KIND = "hybrid-ao-psf-sensing-measurements"


@dataclass(frozen=True, kw_only=True, eq=False)
class WfsSensingStatistics(_PhysicalRecord):
    """Conditional noise and paired local-response estimates at one fixed spot.

    Attributes:
        dtype: Real calculation/state precision, float32 by default or float64.
        mean: Two-component mean measured tilt at zero imposed displacement.
        covariance: Full (2,2) angular-squared population noise covariance.
        white: Dimensionless paired zero draws, stored at the selected
            bank precision, retaining both axes and their sample order.
        gain_matrix: Dimensionless (2,2) local response, output x/y by input x/y.
        gain_standard_error: Dimensionless (2,2) paired-batch standard errors.
        gain_blocks: Dimensionless (blocks,2,2) estimates for independent batches.
        shifted_means: Angular means (output axis,input axis,positive/negative).

    Gain and scalar variance are trace/2 reductions. Array ownership and units
    are immutable. Negative sampled gain remains available for diagnosis;
    building a fitted sensor table requires positive supported gains.
    """

    mean: u.Quantity
    covariance: u.Quantity
    white: u.Quantity
    gain_matrix: u.Quantity
    gain_standard_error: u.Quantity
    gain_blocks: u.Quantity
    shifted_means: u.Quantity

    dtype: DTypeLike = field(default=np.float32, kw_only=True)

    def __post_init__(self):
        object.__setattr__(self, "dtype", validate_psf_dtype(self.dtype))
        values = {
            name: _quantity(getattr(self, name), unit, name, ndim, dtype=self.dtype)
            for name, unit, ndim in (
                ("mean", u.mas, 1),
                ("covariance", u.mas**2, 2),
                ("white", u.one, 2),
                ("gain_matrix", u.one, 2),
                ("gain_standard_error", u.one, 2),
                ("gain_blocks", u.one, 3),
                ("shifted_means", u.mas, 3),
            )
        }
        if (
            values["mean"].shape != (2,)
            or values["covariance"].shape != (2, 2)
            or values["white"].shape[1] != 2
            or values["gain_matrix"].shape != (2, 2)
            or values["gain_standard_error"].shape != (2, 2)
            or values["gain_blocks"].shape[1:] != (2, 2)
            or values["shifted_means"].shape != (2, 2, 2)
        ):
            raise ValueError("invalid two-axis sensing statistic shapes")
        for name, value in values.items():
            object.__setattr__(self, name, value)

    @property
    def gain(self) -> float:
        """Return the scalar local gain, trace(gain_matrix)/2."""
        return float(np.trace(self.gain_matrix.value) / 2)

    @property
    def variance(self) -> u.Quantity:
        """Return isotropic per-axis noise variance, trace(covariance)/2."""
        return np.trace(self.covariance) / 2


def estimate_sensing_statistics(
    samples: SensingWfsMeasurements,
    *,
    displacement: u.Quantity,
    block_draws: int,
    dtype=np.float32,
) -> WfsSensingStatistics:
    """Estimate gain/noise while preserving signed pairing and population divisors.

    Remove the mean conditionally at this fixed input for the noise covariance,
    retaining paired x/y draws and +x,-x,+y,-y ordering. All reductions use
    the selected precision; whitened samples retain that precision.
    """
    if not isinstance(samples, SensingWfsMeasurements):
        raise TypeError("samples must be SensingWfsMeasurements")
    dtype = validate_psf_dtype(dtype)
    offset = _quantity(displacement, u.mas, "displacement", 0, dtype=dtype)
    stats = measurement_statistics(
        samples.zero.value,
        samples.shifted.value,
        float(offset.value),
        block_draws,
        dtype=dtype,
    )
    return WfsSensingStatistics(
        dtype=dtype,
        mean=stats["mean"] * u.mas,
        covariance=stats["covariance"] * u.mas**2,
        white=stats["white"] * u.one,
        gain_matrix=stats["gain_matrix"] * u.one,
        gain_standard_error=stats["gain_se"] * u.one,
        gain_blocks=stats["paired_gain_blocks"] * u.one,
        shifted_means=stats["shifted_means"] * u.mas,
    )


def _sensing_coordinates(position, magnitude, *, dtype=np.float64):
    """Validate one physical WFS direction and scalar magnitude before dispatch."""
    position = _quantity(position, u.arcsec, "position", 1, dtype=dtype)
    magnitude = _quantity(magnitude, u.mag, "magnitude", 0, dtype=dtype)
    if position.shape != (2,):
        raise ValueError("two-axis sensing position required")
    return position, magnitude


@dataclass(frozen=True, kw_only=True, eq=False)
class SensingCalibrationMeasurements(_PhysicalRecord):
    """Reusable paired WFS draws and physical coordinates for one calibration point.

    Attributes:
        dtype: Real calculation/state precision, float32 by default or float64.
        zenith_angle: Scalar physical viewing angle.
        position: Cartesian angular x/y direction, shape (2,).
        magnitude: Scalar magnitude in the declared photometric convention.
        samples: Zero and signed-displacement angular WFS draws, retaining means.
        displacement: Positive scalar angular displacement amplitude.
        sampling: Effective native draw/batch counts and point seed.
        controller: Fixed delayed-integrator settings for subsequent replay.
        photometry: Explicit incident-count conversion for this measured magnitude.

    No native setup or image is needed for offline statistics or rebuilding.
    """

    zenith_angle: u.Quantity
    position: u.Quantity
    magnitude: u.Quantity
    samples: SensingWfsMeasurements
    displacement: u.Quantity
    sampling: SensingWfsSampling
    controller: FixedGainIntegrator
    photometry: CalibrationPhotometry

    dtype: DTypeLike = field(default=np.float32, kw_only=True)

    def __post_init__(self):
        object.__setattr__(self, "dtype", validate_psf_dtype(self.dtype))
        position, magnitude = _sensing_coordinates(
            self.position, self.magnitude, dtype=self.dtype
        )
        values = {
            "zenith_angle": _angle(self.zenith_angle, dtype=self.dtype),
            "position": position,
            "magnitude": magnitude,
            "displacement": _quantity(
                self.displacement, u.mas, "displacement", 0, dtype=self.dtype
            ),
        }
        if values["displacement"].value <= 0:
            raise ValueError("positive displacement required")
        if (
            not isinstance(self.samples, SensingWfsMeasurements)
            or not isinstance(self.sampling, SensingWfsSampling)
            or not isinstance(self.controller, FixedGainIntegrator)
            or not isinstance(self.photometry, CalibrationPhotometry)
        ):
            raise TypeError(
                "typed samples, sampling, controller and photometry required"
            )
        if self.samples.zero.shape != (
            self.sampling.zero_draws,
            2,
        ) or self.samples.shifted.shape != (4, self.sampling.gain_draws, 2):
            raise ValueError(
                "sensing sample counts differ from their effective controls"
            )
        for name, value in values.items():
            object.__setattr__(self, name, value)

    def statistics(self) -> WfsSensingStatistics:
        """Calculate conditional noise and paired gain estimates without native I/O."""
        return estimate_sensing_statistics(
            self.samples,
            displacement=self.displacement,
            block_draws=self.sampling.block_draws,
            dtype=self.dtype,
        )


def save_sensing_measurements(
    measurements: SensingCalibrationMeasurements, path: str | Path
) -> None:
    """Atomically retain zero/signed draws, coordinates and effective controls."""
    if not isinstance(measurements, SensingCalibrationMeasurements):
        raise TypeError("unsupported sensing measurement type")
    header = {
        "schema": _KIND,
        "version": 2,
        "precision": measurements.dtype.name,
        "units": {
            "position": "arcsec",
            "magnitude": "mag",
            "zenith": "deg",
            "samples": "mas",
            "displacement": "mas",
        },
        "zenith_angle_deg": float(measurements.zenith_angle.value),
        "position_arcsec": measurements.position.value.tolist(),
        "magnitude": float(measurements.magnitude.value),
        "displacement_mas": float(measurements.displacement.value),
        "sampling": measurements.sampling.__dict__,
        "controller": controller_header(measurements.controller),
        "photometry": photometry_header(measurements.photometry),
    }
    with atomic_output_path(Path(path)) as temporary, temporary.open("wb") as stream:
        np.savez_compressed(
            stream,
            __header__=json.dumps(header, allow_nan=False),
            zero=measurements.samples.zero.value,
            shifted=measurements.samples.shifted.value,
        )


def load_sensing_measurements(path: str | Path) -> SensingCalibrationMeasurements:
    """Load strict physical WFS observations without any E2E files/software."""
    with np.load(Path(path), allow_pickle=False) as saved:
        header = json.loads(str(saved["__header__"]))
        validate_observation_precision(header, saved)
        if (
            (set(header) - ({"precision"} if header.get("version") == 2 else set()))
            != {
                "schema",
                "version",
                "units",
                "zenith_angle_deg",
                "position_arcsec",
                "magnitude",
                "displacement_mas",
                "sampling",
                "controller",
                "photometry",
            }
            or header["schema"] != _KIND
            or type(header["version"]) is not int
            or header["version"] not in (1, 2)
            or header["units"]
            != {
                "position": "arcsec",
                "magnitude": "mag",
                "zenith": "deg",
                "samples": "mas",
                "displacement": "mas",
            }
            or set(saved.files) != {"__header__", "zero", "shifted"}
        ):
            raise ValueError("invalid sensing measurement schema, units or fields")
        return SensingCalibrationMeasurements(
            dtype=header.get("precision", np.float64),
            zenith_angle=header["zenith_angle_deg"] * u.deg,
            position=header["position_arcsec"] * u.arcsec,
            magnitude=header["magnitude"] * u.mag,
            displacement=header["displacement_mas"] * u.mas,
            samples=SensingWfsMeasurements(
                dtype=header.get("precision", np.float64),
                zero=saved["zero"] * u.mas,
                shifted=saved["shifted"] * u.mas,
            ),
            sampling=SensingWfsSampling(**header["sampling"]),
            controller=read_controller_header(
                header["controller"], dtype=header.get("precision", np.float64)
            ),
            photometry=read_photometry_header(
                header["photometry"], dtype=header.get("precision", np.float64)
            ),
        )
