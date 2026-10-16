"""Physical input and observation contracts for native calibration hooks."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from astropy import units as u


def _quantity(value, unit, name, ndim):
    if not isinstance(value, u.Quantity):
        raise TypeError(f"{name} must be an Astropy Quantity")
    array = np.asarray(value.to_value(unit))
    if (
        array.dtype.kind not in "fiu"
        or array.ndim != ndim
        or not np.isfinite(array).all()
    ):
        raise ValueError(f"{name} must be finite real {ndim}D values")
    if any(n == 0 for n in array.shape):
        raise ValueError(f"{name} must not be empty")
    array = np.asarray(array, dtype=np.float64, order="C")
    immutable = np.frombuffer(array.tobytes(), dtype=np.float64).reshape(array.shape)
    return u.Quantity(immutable, unit, copy=False)


def _count(value, name, minimum=1):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def _angle(value):
    angle = _quantity(value, u.deg, "zenith_angle", 0)
    if not 0 <= angle.value < 90:
        raise ValueError("zenith_angle must lie in [0,90) degrees")
    return angle


class _PhysicalRecord:
    """Give callers quantity views, protecting owned unit metadata and values.

    Stored arrays have immutable byte backing. Each public quantity access
    returns a lightweight view: changing that view's unit cannot alter the
    record, and values cannot become writable through a NumPy flag change.
    """

    def __getattribute__(self, name):
        value = object.__getattribute__(self, name)
        return value.view() if isinstance(value, u.Quantity) else value


@dataclass(frozen=True, kw_only=True, eq=False)
class FixedGainIntegrator(_PhysicalRecord):
    """Fixed delayed-integrator settings shared by calibration and prediction.

    Attributes:
        gain: Positive dimensionless gain, fixed while filtering.
        delay: Positive integer measurement-to-command delay, in frames.
        frame_rate: Positive scalar frequency; one telemetry sample per frame.
    """

    gain: float
    delay: int
    frame_rate: u.Quantity

    def __post_init__(self):
        if isinstance(self.gain, bool) or not np.isfinite(self.gain) or self.gain <= 0:
            raise ValueError("gain must be positive and finite")
        _count(self.delay, "delay")
        rate = _quantity(self.frame_rate, u.Hz, "frame_rate", 0)
        if rate.value <= 0:
            raise ValueError("frame_rate must be positive")
        object.__setattr__(self, "gain", float(self.gain))
        object.__setattr__(self, "frame_rate", rate)

    def transition(self, response: np.ndarray) -> np.ndarray:
        """Return the stable two-axis delayed-command transition matrix.

        response is the finite dimensionless 2-by-2 reconstructed sensing
        matrix. History blocks run newest to oldest. Unstable loops raise
        ValueError; the returned array is newly owned by the caller.
        """
        from ._calibration_numerics import integrator_transition

        return integrator_transition(response, self.gain, self.delay)

    def filter(
        self, history: u.Quantity, response: np.ndarray, *, equilibrium: bool = False
    ) -> u.Quantity:
        """Replay delayed inputs independently across realizations.

        history has shape (realizations,frames,2), angular or OPD units.
        response is the dimensionless 2-by-2 feedback matrix. Return commands
        in the same units and shape. The default has zero command/input prehistory;
        equilibrium=True instead
        holds each first input constant before the first frame. No centring or
        settling cut is applied. Coupled responses are retained exactly, including
        nonsymmetric terms.
        """
        from ._calibration_numerics import filter_input

        if not isinstance(history, u.Quantity):
            raise TypeError("history must be an Astropy quantity")
        values = np.asarray(history.value)
        if (
            values.ndim != 3
            or values.shape[-1] != 2
            or not np.isfinite(values).all()
            or not values.size
        ):
            raise ValueError("history must be finite nonempty (realizations,frames,2)")
        if not (history.unit.is_equivalent(u.rad) or history.unit.is_equivalent(u.m)):
            raise ValueError("history must have angular or OPD units")
        if values.dtype.kind not in "fiu" or type(equilibrium) is not bool:
            raise ValueError("history must be real and equilibrium must be boolean")
        initial = (
            np.linalg.solve(response, values[:, 0, :].T).T
            if equilibrium
            else np.zeros((len(values), 2))
        )
        inputs = values - values[:, :1, :] if equilibrium else values
        result = filter_input(
            inputs.transpose(2, 0, 1), response, self.gain, self.delay
        )
        result = result + initial.T[:, :, None]
        return result.transpose(1, 2, 0) * history.unit

    def noise_covariance(
        self, response: np.ndarray, reconstructed_noise: u.Quantity
    ) -> u.Quantity:
        """Solve the stationary Lyapunov command covariance for white input noise.

        reconstructed_noise is a finite symmetric PSD 2-by-2 angular-squared
        quantity, independent between frames. Return command covariance in
        mas squared. The supplied response is the actual feedback matrix.
        """
        from ._calibration_numerics import integrator_noise_covariance

        if not isinstance(reconstructed_noise, u.Quantity):
            raise TypeError("reconstructed_noise must be an Astropy quantity")
        return (
            integrator_noise_covariance(
                response, reconstructed_noise.to_value(u.mas**2), self.gain, self.delay
            )
            * u.mas**2
        )


@dataclass(frozen=True, kw_only=True, eq=False)
class CalibrationPhotometry(_PhysicalRecord):
    """Numerical magnitude-to-incident-photon conversion supplied downstream.

    Attributes:
        magnitude_band: Explicit calibration magnitude convention, e.g. R.
        photons_per_frame_zeropoint: Positive zero-magnitude incident photon
            count per frame, including the native photon budget before cropping.
    """

    magnitude_band: str
    photons_per_frame_zeropoint: u.Quantity

    def __post_init__(self):
        if not isinstance(self.magnitude_band, str) or not self.magnitude_band.strip():
            raise ValueError("magnitude_band must be a nonempty convention name")
        count = _quantity(
            self.photons_per_frame_zeropoint, u.photon, "photons_per_frame_zeropoint", 0
        )
        if count.value <= 0:
            raise ValueError("photons_per_frame_zeropoint must be positive")
        object.__setattr__(self, "photons_per_frame_zeropoint", count)

    def photons_per_frame(self, magnitudes: u.Quantity) -> u.Quantity:
        """Return incident counts for finite magnitude values in this convention."""
        if not isinstance(magnitudes, u.Quantity):
            raise TypeError("magnitudes must be an Astropy Quantity")
        values = np.asarray(magnitudes.to_value(u.mag), float)
        if not np.isfinite(values).all():
            raise ValueError("magnitudes must be finite")
        result = self.photons_per_frame_zeropoint * 10.0 ** (-0.4 * values)
        if not np.isfinite(result.value).all() or np.any(result.value <= 0):
            raise ValueError("magnitude conversion must produce positive finite counts")
        return result


@dataclass(frozen=True, kw_only=True, eq=False)
class CovarianceTelemetry(_PhysicalRecord):
    """Aligned uncentred histories on one zenith plane, with independent resets.

    Attributes:
        zenith_angle: Scalar viewing angle, in [0,90) degrees.
        positions: Distinct Cartesian field coordinates, (directions,2).
        frame_rate: Positive scalar cadence; every sample is one LO frame.
        tilt: Atmospheric tilt, (realizations,frames,directions,2), angular.
        coma: Aligned primary-coma OPD coefficients, same shape, length.
            The unit-pupil-RMS basis is orthogonal to piston and tilt, after
            common HO correction and before science-specific correction.

    Means are retained. The native reader verifies wavefront stage/basis and
    frame alignment. No realization labels or native setup are transported.
    """

    zenith_angle: u.Quantity
    positions: u.Quantity
    frame_rate: u.Quantity
    tilt: u.Quantity
    coma: u.Quantity

    def __post_init__(self):
        for name, unit, ndim in [
            ("positions", u.arcsec, 2),
            ("frame_rate", u.Hz, 0),
            ("tilt", u.mas, 4),
            ("coma", u.nm, 4),
        ]:
            object.__setattr__(
                self, name, _quantity(getattr(self, name), unit, name, ndim)
            )
        object.__setattr__(self, "zenith_angle", _angle(self.zenith_angle))
        if self.positions.shape[1] != 2 or len(
            np.unique(self.positions.value, axis=0)
        ) != len(self.positions):
            raise ValueError("positions must be distinct (directions,2) pairs")
        if self.frame_rate.value <= 0:
            raise ValueError("frame_rate must be positive")
        if (
            self.tilt.shape[1] < 3
            or self.tilt.shape[2:] != self.positions.shape
            or self.coma.shape != self.tilt.shape
        ):
            raise ValueError(
                "tilt/coma must align as (realizations,frames,directions,2), with >=3 frames"
            )


@dataclass(frozen=True, kw_only=True, eq=False)
class ComaWavefronts(_PhysicalRecord):
    """Paired independent residual-HO samples for native sensor measurement.

    Attributes:
        wavefront: OPD length, (samples,height,width), piston/TT removed and
            zero outside pupil; after common HO, before science correction.
        coma: Primary-coma OPD length, (samples,2), in the telemetry basis.
        pupil: Nonempty boolean (height,width) mask; rows/columns are +y/+x.
    """

    wavefront: u.Quantity
    coma: u.Quantity
    pupil: np.ndarray

    def __post_init__(self):
        wavefront = _quantity(self.wavefront, u.nm, "wavefront", 3)
        coma = _quantity(self.coma, u.nm, "coma", 2)
        pupil = np.asarray(self.pupil)
        if pupil.dtype != bool or pupil.ndim != 2 or not pupil.any():
            raise ValueError("pupil must be a nonempty 2D boolean mask")
        if (
            wavefront.shape[1:] != pupil.shape
            or coma.shape != (len(wavefront), 2)
            or np.any(wavefront.value[:, ~pupil] != 0)
        ):
            raise ValueError("wavefront, coma and pupil must be aligned")
        pupil = np.frombuffer(pupil.tobytes(), dtype=bool).reshape(pupil.shape)
        for name, value in [("wavefront", wavefront), ("coma", coma), ("pupil", pupil)]:
            object.__setattr__(self, name, value)


@dataclass(frozen=True, kw_only=True, eq=False)
class ComaWfsSampling(_PhysicalRecord):
    """Magnitude grid and paired noise draws for a native coma measurement.

    Attributes:
        magnitudes: Strictly increasing magnitude grid with at least two nodes.
        noise_draws: Draws per wavefront/magnitude, a positive multiple of four.
        seed: Nonnegative uint32 seed for deterministic native draw scheduling.
    """

    magnitudes: u.Quantity
    noise_draws: int = 512
    seed: int = 0

    def __post_init__(self):
        values = _quantity(self.magnitudes, u.mag, "magnitudes", 1)
        if len(values) < 2 or np.any(np.diff(values.value) <= 0):
            raise ValueError("magnitudes must be strictly increasing, with >=2 nodes")
        _count(self.noise_draws, "noise_draws", 4)
        _count(self.seed, "seed", 0)
        if self.noise_draws % 4 or self.seed >= 2**32:
            raise ValueError(
                "noise_draws must split into four batches; seed must be uint32"
            )
        object.__setattr__(self, "magnitudes", values)


@dataclass(frozen=True, kw_only=True, eq=False)
class ComaWfsMeasurements(_PhysicalRecord):
    """Uncentred angular WFS observations paired with supplied wavefronts.

    Attributes:
        measured: Tilt (samples,magnitudes,draws,2), physical x/y.
        noiseless: Comparison tilt (samples,magnitudes,2).
    """

    measured: u.Quantity
    noiseless: u.Quantity

    def __post_init__(self):
        measured = _quantity(self.measured, u.mas, "measured", 4)
        noiseless = _quantity(self.noiseless, u.mas, "noiseless", 3)
        if measured.shape[-1] != 2 or noiseless.shape != measured.shape[:2] + (2,):
            raise ValueError("measured/noiseless WFS tilt axes must align")
        object.__setattr__(self, "measured", measured)
        object.__setattr__(self, "noiseless", noiseless)


@dataclass(frozen=True, kw_only=True, eq=False)
class SensingSpot(_PhysicalRecord):
    """Known fixed HO spot and signed displacements for native WFS draws.

    Attributes:
        image: Nonnegative (y,x) fractions per incident photon, positive sum;
            do not recenter or normalize detector cropping.
        pixel_scale: Positive angular scale per image pixel.
        wavelength: Positive scalar sensing wavelength.
        photons_per_frame: Positive incident photon count.
        displacements: Angular (5,2) rows: zero,+x,-x,+y,-y, symmetric
            about zero with the same positive amplitude along both axes.
    """

    image: np.ndarray
    pixel_scale: u.Quantity
    wavelength: u.Quantity
    photons_per_frame: u.Quantity
    displacements: u.Quantity

    def __post_init__(self):
        image = np.asarray(self.image)
        if (
            image.dtype.kind not in "fiu"
            or image.ndim != 2
            or min(image.shape) < 2
            or not np.isfinite(image).all()
            or np.any(image < 0)
            or image.sum() <= 0
        ):
            raise ValueError("image must be a positive finite 2D light-fraction image")
        image = np.asarray(image, dtype=np.float64, order="C")
        object.__setattr__(
            self,
            "image",
            np.frombuffer(image.tobytes(), dtype=np.float64).reshape(image.shape),
        )
        for name, unit in [
            ("pixel_scale", u.mas),
            ("wavelength", u.m),
            ("photons_per_frame", u.photon),
        ]:
            value = _quantity(getattr(self, name), unit, name, 0)
            if value.value <= 0:
                raise ValueError(f"{name} must be positive")
            object.__setattr__(self, name, value)
        offsets = _quantity(self.displacements, u.mas, "displacements", 2)
        if offsets.shape != (5, 2):
            raise ValueError("displacements must have shape (5,2)")
        d = offsets.value[1, 0]
        if d <= 0 or not np.array_equal(
            offsets.value, [[0, 0], [d, 0], [-d, 0], [0, d], [0, -d]]
        ):
            raise ValueError(
                "displacements must be zero,+x,-x,+y,-y with equal amplitudes"
            )
        object.__setattr__(self, "displacements", offsets)


@dataclass(frozen=True, kw_only=True)
class SensingWfsSampling:
    """Counts and reproducibility controls for one fixed-spot measurement.

    Attributes:
        zero_draws: Independent draws at zero imposed displacement.
        gain_draws: Draws at each signed displacement, paired between signs.
        block_draws: Equal batch length, dividing both draw counts.
        seed: Nonnegative uint31-range seed for native scheduling.
    """

    zero_draws: int = 4096
    gain_draws: int = 16384
    block_draws: int = 1024
    seed: int = 261013700

    def __post_init__(self):
        for name in ("zero_draws", "gain_draws", "block_draws"):
            _count(getattr(self, name), name, 4)
        _count(self.seed, "seed", 0)
        if (
            self.zero_draws % self.block_draws
            or self.gain_draws % self.block_draws
            or self.gain_draws < 4 * self.block_draws
            or self.seed >= 2**31
        ):
            raise ValueError(
                "draws must split into complete batches, with >=4 gain batches and uint31 seed"
            )


@dataclass(frozen=True, kw_only=True, eq=False)
class SensingWfsMeasurements(_PhysicalRecord):
    """All native angular samples, retaining means and signed-draw pairing.

    Attributes:
        zero: Unshifted samples (zero_draws,2).
        shifted: Samples (4,gain_draws,2), ordered +x,-x,+y,-y.
    """

    zero: u.Quantity
    shifted: u.Quantity

    def __post_init__(self):
        zero = _quantity(self.zero, u.mas, "zero", 2)
        shifted = _quantity(self.shifted, u.mas, "shifted", 3)
        if zero.shape[1] != 2 or shifted.shape[0] != 4 or shifted.shape[-1] != 2:
            raise ValueError("need zero (draws,2), shifted (4,draws,2)")
        object.__setattr__(self, "zero", zero)
        object.__setattr__(self, "shifted", shifted)


@dataclass(frozen=True, kw_only=True, eq=False)
class ReconstructorNoiseInputs(_PhysicalRecord):
    """Ordered pre-filter noise-query samples, without controller settings.

    Attributes:
        photons_per_frame: Positive photon-count vector.
        ee: Dimensionless fractions, aligned, in (0,1].
        fwhm: Positive angular full widths at half maximum, aligned.
    """

    photons_per_frame: u.Quantity
    ee: u.Quantity
    fwhm: u.Quantity

    def __post_init__(self):
        for name, unit in [
            ("photons_per_frame", u.photon),
            ("ee", u.one),
            ("fwhm", u.mas),
        ]:
            value = _quantity(getattr(self, name), unit, name, 1)
            if np.any(value.value <= 0):
                raise ValueError(f"{name} must be positive")
            object.__setattr__(self, name, value)
        if (
            self.ee.shape != self.fwhm.shape
            or self.ee.shape != self.photons_per_frame.shape
            or np.any(self.ee.value > 1)
        ):
            raise ValueError("noise inputs must align with EE in (0,1]")


@dataclass(frozen=True, kw_only=True, eq=False)
class ReconstructorNoiseMeasurements(_PhysicalRecord):
    """Native pre-filter noise results, distinct from measured WFS variance.

    Attributes:
        covariance: Ordered (samples,2,2) symmetric PSD angular covariances.
    """

    covariance: u.Quantity

    def __post_init__(self):
        covariance = _quantity(self.covariance, u.mas**2, "covariance", 3)
        if covariance.shape[1:] != (2, 2) or not np.allclose(
            covariance.value, covariance.value.swapaxes(-1, -2), rtol=1e-12, atol=0
        ):
            raise ValueError("covariance must be symmetric (samples,2,2)")
        if np.any(np.linalg.eigvalsh(covariance.value) < 0):
            raise ValueError("covariance must be positive semidefinite")
        object.__setattr__(self, "covariance", covariance)


@dataclass(frozen=True, kw_only=True, eq=False)
class ComaResponseMeasurements(_PhysicalRecord):
    """Sufficient paired observations for offline response and joint-law fitting.

    Attributes:
        coma: OPD coefficients (samples,2), uncentred.
        magnitudes: Increasing grid (magnitudes,).
        tilt_mean: Angular noise means (samples,magnitudes,2).
        tilt_batches: Angular means (samples,magnitudes,batches,2), equal-sized
            independent detector batches whose average is tilt_mean.
        tilt_noiseless: Angular comparisons, same shape as tilt_mean.
    """

    coma: u.Quantity
    magnitudes: u.Quantity
    tilt_mean: u.Quantity
    tilt_batches: u.Quantity
    tilt_noiseless: u.Quantity

    def __post_init__(self):
        for name, unit, ndim in [
            ("coma", u.nm, 2),
            ("magnitudes", u.mag, 1),
            ("tilt_mean", u.mas, 3),
            ("tilt_batches", u.mas, 4),
            ("tilt_noiseless", u.mas, 3),
        ]:
            object.__setattr__(
                self, name, _quantity(getattr(self, name), unit, name, ndim)
            )
        n, m = len(self.coma), len(self.magnitudes)
        if (
            n < 4
            or n % 2
            or self.coma.shape != (n, 2)
            or m < 2
            or np.any(np.diff(self.magnitudes.value) <= 0)
            or self.tilt_mean.shape != (n, m, 2)
            or self.tilt_noiseless.shape != (n, m, 2)
            or self.tilt_batches.shape[:2] != (n, m)
            or self.tilt_batches.shape[-1] != 2
            or self.tilt_batches.shape[2] < 2
        ):
            raise ValueError("invalid paired response observation shapes")
        tolerance = 64 * np.spacing(max(1.0, np.max(np.abs(self.tilt_mean.value))))
        if (
            np.max(np.abs(self.tilt_batches.value.mean(axis=2) - self.tilt_mean.value))
            > tolerance
        ):
            raise ValueError("tilt_batches must average to tilt_mean")
