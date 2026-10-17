"""Primary-coma response calibration, with measured support and physical units."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
from astropy import units as u
from numpy.typing import DTypeLike

from ._calibration_config import ComaCalibrationSettings
from ._calibration_contracts import (
    CalibrationPhotometry,
    ComaResponseMeasurements,
    _angle,
    _PhysicalRecord,
    _quantity,
)
from ._calibration_io import validate_observation_precision
from ._calibration_products import (
    immutable_metadata,
    load_product,
    photometry_header,
    save_product,
)
from ._coma_fitting import _ResponseArrays, fit_airmass, fit_response
from ._io import atomic_output_path
from ._validation import clip_support, support_dtype, validate_psf_dtype

_KIND = "hybrid-ao-psf-coma-response"
_MEASUREMENTS = "hybrid-ao-psf-coma-measurements"
_FIELDS = ("coma", "magnitudes", "tilt_mean", "tilt_batches", "tilt_noiseless")
_UNITS = {
    "coma": "nm",
    "magnitudes": "mag",
    "tilt_mean": "mas",
    "tilt_batches": "mas",
    "tilt_noiseless": "mas",
}


@dataclass(frozen=True, kw_only=True, eq=False)
class ComaCalibrationMeasurements(_PhysicalRecord):
    """Reusable paired response observations and their effective fit controls.

    Attributes:
        dtype: Real calculation/state precision, float32 by default or float64.
        zenith_angle: Physical viewing angle of these independent wavefronts.
        response: Primary-coma OPD and uncentred mean/batch/noiseless WFS tilt.
        sensing_wavelength: Positive WFS reference wavelength. It preserves the
            retained optimizer's phase-to-OPD numerical conditioning only;
            fitted response and prediction always use mas/nm OPD.
        photometry: Measured magnitude convention and incident-count zero point.
        settings: Effective sample/noise-draw controls and bootstrap seed.

    No native setup, provenance dictionary or acquisition file is required by
    the standalone estimator. Means remain intact until paired regression.
    """

    zenith_angle: u.Quantity
    response: ComaResponseMeasurements
    sensing_wavelength: u.Quantity
    photometry: CalibrationPhotometry
    settings: ComaCalibrationSettings

    dtype: DTypeLike = field(default=np.float32, kw_only=True)

    def __post_init__(self):
        object.__setattr__(self, "dtype", validate_psf_dtype(self.dtype))
        angle = _angle(self.zenith_angle, dtype=self.dtype)
        wavelength = _quantity(
            self.sensing_wavelength, u.m, "sensing_wavelength", 0, dtype=self.dtype
        )
        if wavelength.value <= 0:
            raise ValueError("sensing_wavelength must be positive")
        if (
            not isinstance(self.response, ComaResponseMeasurements)
            or not isinstance(self.photometry, CalibrationPhotometry)
            or not isinstance(self.settings, ComaCalibrationSettings)
        ):
            raise TypeError("typed response, photometry and coma settings required")
        batches = self.response.tilt_batches.shape[2]
        if (
            len(self.response.coma) != self.settings.samples
            or self.settings.noise_draws % batches
        ):
            raise ValueError(
                "response sample/batch axes differ from effective controls"
            )
        object.__setattr__(self, "zenith_angle", angle)
        object.__setattr__(self, "sensing_wavelength", wavelength)


@dataclass(frozen=True, kw_only=True, eq=False)
class ComaResponseModel(_PhysicalRecord):
    """Positive three-parameter magnitude/airmass response over measured support.

    Attributes:
        producer_dtype: Recorded calibration precision, or None when unknown.
        support_precision: Coarsest coordinate precision retained through preparation.
        dtype: Real calculation/state precision, float32 by default or float64.
        a0: Zenith bright-star response in angular tilt per OPD length.
        b: Response change per unit increase in airmass, in the same units.
        m_half: Magnitude at half response, inside the measured magnitude range.
        magnitude_support: Two increasing measured magnitude limits.
        zenith_support: Two increasing measured angular viewing limits.
        photometry: Explicit magnitude convention and incident-count conversion.

    The response is (a0+b*(airmass-1))/(1+10**(0.4*(m-m_half))). No native
    wavelength or E2E software is used at evaluation; coma is already OPD.
    """

    a0: u.Quantity
    b: u.Quantity
    m_half: u.Quantity
    magnitude_support: u.Quantity
    zenith_support: u.Quantity
    photometry: CalibrationPhotometry

    dtype: DTypeLike = field(default=np.float32, kw_only=True)
    producer_dtype: DTypeLike | None = field(default=None, kw_only=True)
    support_precision: DTypeLike | None = field(default=None, kw_only=True)

    def __post_init__(self):
        object.__setattr__(self, "dtype", validate_psf_dtype(self.dtype))
        if self.producer_dtype is not None:
            object.__setattr__(
                self, "producer_dtype", validate_psf_dtype(self.producer_dtype)
            )
        object.__setattr__(
            self,
            "support_precision",
            support_dtype(self.dtype, self.producer_dtype, self.support_precision),
        )
        values = {
            name: _quantity(getattr(self, name), unit, name, ndim, dtype=self.dtype)
            for name, unit, ndim in (
                ("a0", u.mas / u.nm, 0),
                ("b", u.mas / u.nm, 0),
                ("m_half", u.mag, 0),
                ("magnitude_support", u.mag, 1),
                ("zenith_support", u.deg, 1),
            )
        }
        magnitude, angles = (
            values["magnitude_support"].value,
            values["zenith_support"].value,
        )
        if (
            magnitude.shape != (2,)
            or angles.shape != (2,)
            or magnitude[0] >= magnitude[1]
            or not 0 <= angles[0] < angles[1] < 90
        ):
            raise ValueError("increasing measured magnitude/zenith support required")
        if not magnitude[0] < values["m_half"].value < magnitude[1]:
            raise ValueError("half-response magnitude must lie inside measured support")
        amplitude = values["a0"].value + values["b"].value * (
            1 / np.cos(np.deg2rad(angles)) - 1
        )
        if np.min(amplitude) <= 0:
            raise ValueError("coma response must be positive over its measured support")
        if not isinstance(self.photometry, CalibrationPhotometry):
            raise TypeError("photometry must be CalibrationPhotometry")
        for name, value in values.items():
            object.__setattr__(self, name, value)

    def prepare(self, *, dtype=np.float32) -> ComaResponseModel:
        """Copy runtime coefficients while retaining producer precision."""
        return replace(
            self,
            dtype=validate_psf_dtype(dtype),
            photometry=replace(self.photometry, dtype=dtype),
        )

    def predict(self, *, magnitude: u.Quantity, zenith_angle: u.Quantity) -> u.Quantity:
        """Return broadcast primary-coma WFS response in mas/nm OPD.

        Require finite physical magnitude/angle values inside actual measured
        support. The caller uses this model's explicit magnitude convention.
        """
        if not isinstance(magnitude, u.Quantity) or not isinstance(
            zenith_angle, u.Quantity
        ):
            raise TypeError("magnitude and zenith_angle must be physical quantities")
        mag, angle = np.broadcast_arrays(
            np.asarray(magnitude.to_value(u.mag), dtype=self.dtype),
            np.asarray(zenith_angle.to_value(u.deg), dtype=self.dtype),
        )
        if any(
            v.dtype.kind not in "fiu" or not np.isfinite(v).all() for v in (mag, angle)
        ):
            raise ValueError("finite real magnitude/angle values required")
        precision = self.support_precision
        mag = clip_support(
            mag,
            self.magnitude_support.value,
            dtype=precision,
            label="coma-response magnitude",
        )
        angle = clip_support(
            angle,
            self.zenith_support.value,
            dtype=precision,
            label="coma-response zenith angle",
        )
        amplitude = self.a0 + self.b * (1 / np.cos(np.deg2rad(angle)) - 1)
        return (amplitude / (1 + 10 ** (0.4 * (mag - self.m_half.value)))).to(
            u.mas / u.nm
        )


def _diagnostics(value):
    """Represent undefined matrix/half uncertainties as explicit JSON nulls."""
    if isinstance(value, np.ndarray):
        return _diagnostics(value.tolist())
    if isinstance(value, np.generic):
        return _diagnostics(value.item())
    if isinstance(value, dict):
        return {key: _diagnostics(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_diagnostics(item) for item in value]
    if isinstance(value, float) and np.isnan(value):
        return None
    return value


@dataclass(frozen=True, kw_only=True)
class ComaCalibrationResult:
    """Standalone response law and immutable regression/fit diagnostics.

    Attributes:
        model: Accepted response law with bounds from actual measured planes.
        diagnostics: Per-plane scalar/matrix/bootstrap estimates and joint-law
            checks. Array dimensions are retained as tuples; undefined optional
            matrix/half uncertainties are None, never invented finite values.
    """

    model: ComaResponseModel
    diagnostics: Mapping

    def __post_init__(self):
        if not isinstance(self.model, ComaResponseModel):
            raise TypeError("model must be ComaResponseModel")
        object.__setattr__(
            self, "diagnostics", immutable_metadata(_diagnostics(self.diagnostics))
        )


def build_coma_calibration(
    measurements: Iterable[ComaCalibrationMeasurements],
    *,
    heldout_zenith: u.Quantity | None = 30 * u.deg,
    bootstrap_seed: int | None = None,
    dtype=np.float32,
) -> ComaCalibrationResult:
    """Estimate each plane and fit the retained joint response law offline.

    Preserve scalar/matrix paired regression, the 200-replicate full/half
    wavefront bootstrap, four detector batches, optimizer initialization and
    acceptance criteria. Inputs must share magnitude nodes, photometry and
    conditioning wavelength. At least three distinct viewing planes are needed;
    optional held-out validation runs only with three other training planes.
    bootstrap_seed=None retains each recorded fit seed; an explicit uint32
    seed changes resampling without reacquiring WFS observations. No wavefront
    images, native files or calibration subclass are required.
    """
    if bootstrap_seed is not None:
        from ._calibration_contracts import _count

        _count(bootstrap_seed, "bootstrap_seed", 0)
        if bootstrap_seed >= 2**32:
            raise ValueError("bootstrap_seed must fit uint32")
    measurements = list(measurements)
    if len(measurements) < 3 or any(
        not isinstance(item, ComaCalibrationMeasurements) for item in measurements
    ):
        raise ValueError("at least three typed coma measurement planes required")
    dtype = validate_psf_dtype(dtype)
    coordinate_precision = support_dtype(dtype, *(item.dtype for item in measurements))
    measurements = [
        replace(
            item,
            dtype=dtype,
            response=replace(item.response, dtype=dtype),
            photometry=replace(item.photometry, dtype=dtype),
        )
        for item in measurements
    ]
    measurements.sort(key=lambda item: float(item.zenith_angle.value))
    first = measurements[0]
    angles = np.array([item.zenith_angle.value for item in measurements], dtype=dtype)
    magnitudes = first.response.magnitudes.value
    if len(np.unique(angles)) != len(angles):
        raise ValueError("duplicate coma zenith planes")
    fits = []
    for item in measurements:
        if (
            not np.array_equal(item.response.magnitudes.value, magnitudes)
            or item.sensing_wavelength != first.sensing_wavelength
            or item.photometry.magnitude_band != first.photometry.magnitude_band
            or item.photometry.photons_per_frame_zeropoint
            != first.photometry.photons_per_frame_zeropoint
        ):
            raise ValueError(
                "incompatible coma magnitude, photometry or conditioning conventions"
            )
        response = _ResponseArrays(
            **{name: getattr(item.response, name).value for name in _FIELDS}
        )
        fit = fit_response(
            response,
            sensing_wavelength=float(item.sensing_wavelength.value),
            bootstrap_seed=item.settings.bootstrap_seed
            if bootstrap_seed is None
            else bootstrap_seed,
            noise_draws=item.settings.noise_draws,
        )
        if not fit["accepted"]:
            raise ValueError(
                f"rejected coma response estimate at zenith {item.zenith_angle}"
            )
        fits.append(fit)
    heldout = -1.0 if heldout_zenith is None else float(_angle(heldout_zenith).value)
    joint = fit_airmass(
        angles,
        magnitudes,
        np.stack([fit["h"] for fit in fits]),
        np.stack([fit["standardError"] for fit in fits]),
        sensing_wavelength=float(first.sensing_wavelength.value),
        heldout_zenith=heldout,
    )
    if not joint["accepted"]:
        raise ValueError("joint coma response law failed acceptance")
    a0, b, half = joint["parameters"]
    model = ComaResponseModel(
        dtype=dtype,
        producer_dtype=dtype,
        support_precision=coordinate_precision,
        a0=a0 * u.mas / u.nm,
        b=b * u.mas / u.nm,
        m_half=half * u.mag,
        magnitude_support=magnitudes[[0, -1]] * u.mag,
        zenith_support=angles[[0, -1]] * u.deg,
        photometry=first.photometry,
    )
    return ComaCalibrationResult(
        model=model,
        diagnostics={
            "planes": {
                f"{angle:g}": fit for angle, fit in zip(angles, fits, strict=True)
            },
            "joint": joint,
            "bootstrap_seeds": [
                item.settings.bootstrap_seed
                if bootstrap_seed is None
                else bootstrap_seed
                for item in measurements
            ],
        },
    )


def save_coma_calibration(model: ComaResponseModel, path: str | Path) -> None:
    """Atomically publish a JSON-only response model with physical support."""
    if not isinstance(model, ComaResponseModel):
        raise TypeError("model must be ComaResponseModel")
    save_product(
        path,
        _KIND,
        {
            "precision": model.dtype.name,
            "support_precision": model.support_precision.name,
            "producer_precision": None
            if model.producer_dtype is None
            else np.dtype(model.producer_dtype).name,
            "parameters": {
                "a0_mas_per_nm": float(model.a0.value),
                "b_mas_per_nm": float(model.b.value),
                "m_half_mag": float(model.m_half.value),
            },
            "magnitude_support": model.magnitude_support.value.tolist(),
            "zenith_support_deg": model.zenith_support.value.tolist(),
            "photometry": photometry_header(model.photometry),
            "units": {"response": "mas/nm", "magnitude": "mag", "zenith": "deg"},
        },
    )


def load_coma_calibration(path: str | Path) -> ComaResponseModel:
    """Load exactly the shared response schema; never search native output trees."""
    header, arrays = load_product(path, _KIND)
    if (
        arrays is not None
        or (set(header) - {"precision", "producer_precision", "support_precision"})
        != {
            "schema",
            "version",
            "parameters",
            "magnitude_support",
            "zenith_support_deg",
            "photometry",
            "units",
        }
        or header["units"]
        != {"response": "mas/nm", "magnitude": "mag", "zenith": "deg"}
        or set(header["parameters"]) != {"a0_mas_per_nm", "b_mas_per_nm", "m_half_mag"}
        or set(header["photometry"])
        != {"magnitude_band", "photons_per_frame_zeropoint"}
    ):
        raise ValueError("invalid coma model fields/units")
    coefficients = header["parameters"]
    photometry = header["photometry"]
    return ComaResponseModel(
        dtype=header.get("precision", np.float64),
        producer_dtype=header.get("producer_precision"),
        support_precision=header.get("support_precision"),
        a0=coefficients["a0_mas_per_nm"] * u.mas / u.nm,
        b=coefficients["b_mas_per_nm"] * u.mas / u.nm,
        m_half=coefficients["m_half_mag"] * u.mag,
        magnitude_support=header["magnitude_support"] * u.mag,
        zenith_support=header["zenith_support_deg"] * u.deg,
        photometry=CalibrationPhotometry(
            dtype=header.get("precision", np.float64),
            magnitude_band=photometry["magnitude_band"],
            photons_per_frame_zeropoint=photometry["photons_per_frame_zeropoint"]
            * u.photon,
        ),
    )


def save_coma_measurements(
    measurements: ComaCalibrationMeasurements, path: str | Path
) -> None:
    """Atomically save sufficient paired observations and effective numerical controls."""
    if not isinstance(measurements, ComaCalibrationMeasurements):
        raise TypeError("unsupported coma measurement type")
    header = {
        "schema": _MEASUREMENTS,
        "version": 2,
        "precision": measurements.dtype.name,
        "zenith_angle_deg": float(measurements.zenith_angle.value),
        "sensing_wavelength_m": float(measurements.sensing_wavelength.value),
        "photometry": photometry_header(measurements.photometry),
        "settings": measurements.settings.__dict__,
        "units": _UNITS,
    }
    with atomic_output_path(Path(path)) as temporary, temporary.open("wb") as stream:
        np.savez_compressed(
            stream,
            __header__=json.dumps(header, allow_nan=False),
            **{name: getattr(measurements.response, name).value for name in _FIELDS},
        )


def load_coma_measurements(path: str | Path) -> ComaCalibrationMeasurements:
    """Load strict paired response observations without wavefront/native resources."""
    with np.load(Path(path), allow_pickle=False) as saved:
        header = json.loads(str(saved["__header__"]))
        validate_observation_precision(header, saved)
        if (
            (set(header) - ({"precision"} if header.get("version") == 2 else set()))
            != {
                "schema",
                "version",
                "zenith_angle_deg",
                "sensing_wavelength_m",
                "photometry",
                "settings",
                "units",
            }
            or header["schema"] != _MEASUREMENTS
            or type(header["version"]) is not int
            or header["version"] not in (1, 2)
            or header["units"] != _UNITS
            or set(saved.files) != {"__header__", *_FIELDS}
            or set(header["photometry"])
            != {"magnitude_band", "photons_per_frame_zeropoint"}
        ):
            raise ValueError("invalid coma measurements schema/units/fields")
        photometry = header["photometry"]
        return ComaCalibrationMeasurements(
            dtype=header.get("precision", np.float64),
            zenith_angle=header["zenith_angle_deg"] * u.deg,
            response=ComaResponseMeasurements(
                dtype=header.get("precision", np.float64),
                **{name: saved[name] * u.Unit(_UNITS[name]) for name in _FIELDS},
            ),
            sensing_wavelength=header["sensing_wavelength_m"] * u.m,
            photometry=CalibrationPhotometry(
                dtype=header.get("precision", np.float64),
                magnitude_band=photometry["magnitude_band"],
                photons_per_frame_zeropoint=photometry["photons_per_frame_zeropoint"]
                * u.photon,
            ),
            settings=ComaCalibrationSettings(**header["settings"]),
        )
