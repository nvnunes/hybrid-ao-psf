"""Immutable public contracts for one Hybrid AO PSF simulation."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np
from ao_core.stats import PsfMetadata
from astropy import units as u
from numpy.typing import DTypeLike

from ._units import _PhysicalRecord
from ._validation import validate_psf_dtype

if TYPE_CHECKING:
    from ._e2e_prediction import E2ELowOrderPrediction

# Calibrated against released MASTSEL output. Negative eigenvalues at or below
# this magnitude represent a material covariance failure in nm^2.
CTOT_PSD_TOLERANCE_NM2 = 1.0e-9


class LowOrderModel(StrEnum):
    """Low-order model identifier used in configuration and results."""

    MASTSEL = "mastsel"
    E2E_CALIBRATED = "e2e-calibrated"


class NonPsdPolicy(StrEnum):
    """Researcher-selected handling for materially non-PSD Ctot matrices."""

    ERROR = "error"
    CLIP = "clip"


class DiagnosticsLevel(StrEnum):
    """Amount of optional engine data retained with a result."""

    NONE = "none"
    VALIDATION = "validation"
    DEBUG = "debug"


@dataclass(frozen=True)
class PsfPrediction:
    """PSFs returned for requested coordinates.

    Attributes:
        psfs: One non-negative PSF image per requested field coordinate, in
            ``(N, y, x)`` order. Columns increase toward ``+x`` and rows
            toward ``+y`` in the request's Cartesian frame.
        pixel_scale: Scalar angular pixel scale shared by the returned PSFs.
        metadata: Physical PSF metadata compatible with AO Core.
        dtype: PSF pixel precision, float32 by default. Explicit float64 retains
            double-precision source pixels; other precisions are rejected.
    """

    psfs: np.ndarray
    pixel_scale: u.Quantity
    metadata: PsfMetadata
    dtype: DTypeLike = field(default=np.float32, kw_only=True)

    def __post_init__(self) -> None:
        dtype = validate_psf_dtype(self.dtype)
        psfs = _readonly_array(self.psfs, dtype=dtype, label="psfs", ndim=3)
        object.__setattr__(self, "dtype", dtype)
        if np.any(psfs < 0.0):
            raise ValueError("psfs must be non-negative.")
        flux = np.sum(psfs, axis=(-2, -1), dtype=dtype)
        if np.any(flux <= 0.0):
            raise ValueError("psfs must have positive flux.")
        pixel_scale = _positive_scalar(
            self.pixel_scale, u.mas, "pixel_scale", dtype=dtype
        )
        metadata = _readonly_metadata(
            self.metadata, num_psfs=psfs.shape[0], dtype=dtype
        )
        if metadata.pixel_scale.ndim != 0 or not np.isclose(
            metadata.pixel_scale.to_value(u.mas), pixel_scale.to_value(u.mas)
        ):
            raise ValueError("pixel_scale must match scalar metadata.pixel_scale.")
        object.__setattr__(self, "psfs", psfs)
        object.__setattr__(self, "pixel_scale", pixel_scale)
        object.__setattr__(self, "metadata", metadata)


@dataclass(frozen=True)
class PsfMetricPrediction(_PhysicalRecord):
    """PSF metrics with immutable values and protected unit metadata.

    Quantity access returns a view; changing its unit does not alter the
    record. Numeric storage cannot become writable through NumPy flags.

    Attributes:
        dtype: float32 by default or explicit float64. None preserves individual
            input array precisions.
        ee: Encircled-energy fraction for each requested field coordinate.
        fwhm: Full width at half maximum for each requested field coordinate.
        sr: Strehl ratio for each requested field coordinate.
    """

    ee: u.Quantity
    fwhm: u.Quantity
    sr: u.Quantity
    dtype: DTypeLike | None = field(default=np.float32, kw_only=True)

    def __post_init__(self) -> None:
        dtype = None if self.dtype is None else validate_psf_dtype(self.dtype)
        object.__setattr__(self, "dtype", dtype)
        ee = _readonly_quantity(
            self.ee, u.dimensionless_unscaled, "ee", ndim=1, dtype=dtype
        )
        fwhm = _readonly_quantity(self.fwhm, u.mas, "fwhm", ndim=1, dtype=dtype)
        sr = _readonly_quantity(
            self.sr, u.dimensionless_unscaled, "sr", ndim=1, dtype=dtype
        )
        if not (ee.shape == fwhm.shape == sr.shape):
            raise ValueError("PSF metric vectors must have identical shapes.")
        if ee.size == 0:
            raise ValueError("PSF metric vectors must not be empty.")
        if np.any(ee.value < 0.0) or np.any(sr.value < 0.0):
            raise ValueError("ee and sr must be non-negative.")
        if np.any(fwhm.value <= 0.0):
            raise ValueError("fwhm must be positive.")
        object.__setattr__(self, "ee", ee)
        object.__setattr__(self, "fwhm", fwhm)
        object.__setattr__(self, "sr", sr)


@runtime_checkable
class PsfProvider(Protocol):
    """Supply PSFs at requested field coordinates, wavelength, and zenith angle.

    Implementations may read existing images or interpolate samples. They own
    the supported coordinates, image sampling, flux convention, and source
    access. Each evaluation returns independent, read-only data.
    """

    def evaluate(
        self,
        *,
        zenith_angle: u.Quantity,
        wavelength: u.Quantity,
        x: u.Quantity,
        y: u.Quantity,
    ) -> PsfPrediction:
        """Return PSFs at one wavelength and zenith angle for the requested field points.

        Args:
            zenith_angle: Common zenith angle as a scalar angular quantity.
            wavelength: Common wavelength as a scalar length quantity.
            x: Field positions along x as an angular quantity.
            y: Field positions along y as an angular quantity paired with ``x``.

        Returns:
            One image per coordinate pair in request order, with a common angular
            pixel scale and physical metadata. Preserve the provider's source
            flux convention and return rows towards +y and columns towards +x.

        Raises:
            ValueError: If a requested coordinate, wavelength, or zenith angle
                cannot be supplied.
            FileNotFoundError: If the implementation requires an unavailable file.
        """
        ...


@runtime_checkable
class PsfMetricProvider(Protocol):
    """Supply PSF metrics at requested field coordinates and zenith angle.

    Implementations own the source of the values and the coordinates they
    can satisfy. Results align with the requested field order.
    """

    def evaluate(
        self,
        *,
        zenith_angle: u.Quantity,
        x: u.Quantity,
        y: u.Quantity,
    ) -> PsfMetricPrediction:
        """Return PSF metrics for the requested field points at one zenith angle.

        Args:
            zenith_angle: Common zenith angle as a scalar angular quantity.
            x: Field positions along x as an angular quantity.
            y: Field positions along y as an angular quantity paired with ``x``.

        Returns:
            Encircled energy, FWHM and Strehl ratio in request order.

        Raises:
            ValueError: If requested field coordinates or zenith angle cannot
                be supplied.
            FileNotFoundError: If the implementation requires an unavailable file.
        """
        ...


@dataclass(frozen=True, kw_only=True)
class HybridRequest(_PhysicalRecord):
    """Observation and output settings for one Hybrid AO PSF simulation.

    Attributes:
        dtype: Real calculation/state precision, float32 by default or float64.
        science_positions: Angular science coordinates of shape ``(science, 2)``,
            one Cartesian (x, y) pair per row.
        ngs_positions: Angular NGS coordinates of shape ``(NGS, 2)`` in the
            same frame, ordered to match the NGS fluxes or magnitudes.
        ngs_flux: Effective photon rate for each active guide star, as an
            alternative to ``ngs_magnitude`` for models accepting flux.
        ngs_magnitude: Magnitude of each active guide star, when supplying
            magnitudes instead of effective flux.
        wavelength: Science wavelength.
        zenith_angle: Observation zenith angle.
        diagnostics_level: Amount of optional intermediate data to retain.
    """

    science_positions: u.Quantity
    ngs_positions: u.Quantity
    ngs_flux: u.Quantity | None = None
    ngs_magnitude: u.Quantity | None = None
    wavelength: u.Quantity
    zenith_angle: u.Quantity
    diagnostics_level: DiagnosticsLevel | str = DiagnosticsLevel.NONE

    dtype: DTypeLike = field(default=np.float32, kw_only=True)

    def __post_init__(self) -> None:
        dtype = validate_psf_dtype(self.dtype)
        object.__setattr__(self, "dtype", dtype)
        for name in ("science_positions", "ngs_positions"):
            positions = _readonly_quantity(
                getattr(self, name), u.arcsec, name, ndim=2, dtype=dtype
            )
            if positions.shape[1] != 2:
                raise ValueError(f"{name} must have shape (N, 2) in (x, y) order.")
            object.__setattr__(self, name, positions)
        if (self.ngs_flux is None) == (self.ngs_magnitude is None):
            raise ValueError("Supply exactly one of ngs_flux or ngs_magnitude.")
        ngs_flux = None
        ngs_magnitude = None
        if self.ngs_flux is not None:
            ngs_flux = _readonly_quantity(
                self.ngs_flux, u.photon / u.s, "ngs_flux", ndim=1, dtype=dtype
            )
            if np.any(ngs_flux.value <= 0.0):
                raise ValueError("ngs_flux must be positive.")
        else:
            ngs_magnitude = _readonly_quantity(
                self.ngs_magnitude, u.mag, "ngs_magnitude", ndim=1, dtype=dtype
            )
        ngs_input = ngs_flux if ngs_flux is not None else ngs_magnitude
        if len(self.ngs_positions) != len(ngs_input):
            raise ValueError("Supply one NGS flux or magnitude per row of ngs_positions.")
        wavelength = _positive_scalar(self.wavelength, u.um, "wavelength", dtype=dtype)
        zenith_angle = _scalar(self.zenith_angle, u.deg, "zenith_angle", dtype=dtype)
        if not 0.0 <= zenith_angle.to_value(u.deg) < 90.0:
            raise ValueError("zenith_angle must be in [0, 90) deg.")
        try:
            diagnostics_level = DiagnosticsLevel(self.diagnostics_level)
        except ValueError as exc:
            raise ValueError(
                "diagnostics_level must be 'none', 'validation', or 'debug'."
            ) from exc
        object.__setattr__(self, "ngs_flux", ngs_flux)
        object.__setattr__(self, "ngs_magnitude", ngs_magnitude)
        object.__setattr__(self, "wavelength", wavelength)
        object.__setattr__(self, "zenith_angle", zenith_angle)
        object.__setattr__(self, "diagnostics_level", diagnostics_level)


@dataclass(frozen=True)
class HybridMastselStatistics:
    """Read-only snapshots of the baseline's mathematical LO quantities.

    Covariances use nm² and interleaved NGS x/y ordering. No sensor or symbolic
    implementation objects are exposed. This record is transient, not persisted.

    Attributes:
        ngs_covariance: Css, shape ``(2*NGS, 2*NGS)``.
        science_ngs_covariance: Cas, shape ``(Nscience, 2, 2*NGS)``.
        centre_ngs_covariance: Centre Cas, shape ``(2, 2*NGS)``; may be
            absent for one NGS when the science field omits the centre.
        reconstructor: Exact baseline R blocks, dimensionless, shape
            ``(Nscience, 2, 2*NGS)``.
        measurement_noise: Filtered NGS noise-only covariance, shape
            ``(2*NGS, 2*NGS)``; absent for non-fixed controller gain.
    """

    ngs_covariance: u.Quantity
    science_ngs_covariance: u.Quantity
    centre_ngs_covariance: u.Quantity | None
    reconstructor: np.ndarray
    measurement_noise: u.Quantity | None

    def __post_init__(self) -> None:
        css = _readonly_quantity(self.ngs_covariance, u.nm**2, "ngs_covariance", ndim=2)
        cas = _readonly_quantity(
            self.science_ngs_covariance, u.nm**2, "science_ngs_covariance", ndim=3
        )
        r = _readonly_array(
            self.reconstructor, dtype=float, label="reconstructor", ndim=3
        )
        size = css.shape[0]
        if size % 2 or css.shape != (size, size) or cas.shape[1:] != (2, size):
            raise ValueError("MASTSEL covariances require interleaved NGS x/y axes.")
        if r.shape != cas.shape:
            raise ValueError(
                "Reconstructor blocks must match science/NGS covariance shape."
            )
        object.__setattr__(self, "ngs_covariance", css)
        object.__setattr__(self, "science_ngs_covariance", cas)
        object.__setattr__(self, "reconstructor", r)
        for name, expected in (
            ("centre_ngs_covariance", (2, size)),
            ("measurement_noise", (size, size)),
        ):
            value = getattr(self, name)
            if value is not None:
                value = _readonly_quantity(value, u.nm**2, name, ndim=2)
                if value.shape != expected:
                    raise ValueError(f"{name} must have shape {expected}.")
                object.__setattr__(self, name, value)


@dataclass(frozen=True)
class HybridCtotContext:
    """Observation inputs and baseline statistics for a covariance correction.

    Attributes:
        request: Observation request with active NGS and science coordinates.
        ngs_metrics: Interpolated NGS metrics supplied to MASTSEL.
        mastsel_statistics: Captured baseline covariances and reconstructor.
        angle_to_wavefront_scale: Positive scalar conversion in nm/mas.
        lo_frame_rate: Scalar LO WFS sampling rate in Hz.
        loop_gain: Configured fixed gain, or None when no numeric gain is set.
        loop_delay: Configured command delay in frames, or None when no
            integer delay is specified.
        ngs_magnitude_zeropoint: Pre-aperture photon flux used to convert
            magnitudes, or None when effective photon rates were supplied.
    """

    request: HybridRequest
    ngs_metrics: PsfMetricPrediction
    mastsel_statistics: HybridMastselStatistics
    angle_to_wavefront_scale: u.Quantity
    lo_frame_rate: u.Quantity
    loop_gain: float | None
    loop_delay: int | None
    ngs_magnitude_zeropoint: u.Quantity | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.request, HybridRequest):
            raise TypeError("request must be a HybridRequest.")
        if not isinstance(self.ngs_metrics, PsfMetricPrediction):
            raise TypeError("ngs_metrics must be a PsfMetricPrediction.")
        size = len(self.request.ngs_positions)
        if self.ngs_metrics.ee.size != size:
            raise ValueError("NGS metrics must match the request's active NGS.")
        object.__setattr__(
            self,
            "angle_to_wavefront_scale",
            _positive_scalar(
                self.angle_to_wavefront_scale, u.nm / u.mas, "angle_to_wavefront_scale"
            ),
        )
        object.__setattr__(
            self,
            "lo_frame_rate",
            _positive_scalar(self.lo_frame_rate, u.Hz, "lo_frame_rate"),
        )
        statistics = self.mastsel_statistics
        if not isinstance(statistics, HybridMastselStatistics):
            raise TypeError("mastsel_statistics must be HybridMastselStatistics.")
        if statistics.science_ngs_covariance.shape != (
            len(self.request.science_positions),
            2,
            2 * size,
        ):
            raise ValueError(
                "MASTSEL statistics must match the request's field and NGS."
            )


def _freeze_input_files(paths: Mapping[str, str]) -> Mapping[str, str]:
    """Copy a flat mapping of input roles to path strings."""
    if not isinstance(paths, Mapping) or any(
        not isinstance(role, str) or not role or not isinstance(path, str) or not path
        for role, path in paths.items()
    ):
        raise TypeError("input_files must map nonempty strings to file paths.")
    return MappingProxyType(dict(paths))


@dataclass(frozen=True)
class HybridCtotCorrection:
    """A field-wide covariance callback with optional input file paths.

    Attributes:
        apply: Callable ``apply(ctot, context)`` receiving unconditioned
            ``(Nscience, 2, 2)`` baseline Ctot in nm² and the correction context.
            Return a replacement Quantity convertible to nm² with identical
            shape/order, or None for no change. Do not mutate the inputs.
            Exceptions propagate; conditioning and the single blur stay in
            the engine.
        input_files: Paths to files used by the correction, keyed by role.
            The correction owns this mapping.
    """

    apply: Callable[[u.Quantity, HybridCtotContext], u.Quantity | None]
    input_files: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not callable(self.apply):
            raise TypeError("apply must be callable.")
        object.__setattr__(self, "input_files", _freeze_input_files(self.input_files))


@dataclass(frozen=True)
class CovarianceDiagnostics(_PhysicalRecord):
    """Applied displacement covariance and its PSD checks.

    Attributes:
        ctot: Total angular covariance in mas squared, with shape ``(N, 2, 2)``.
        psd_valid_mask: Mask of nonzero PSD covariance matrices in field order.
        psd_clipped: Mask of positions where MASTSEL clipped a negative
            eigenvalue. None for E2E-Calibrated.
    """

    ctot: u.Quantity
    psd_valid_mask: np.ndarray
    psd_clipped: np.ndarray | None = None

    def __post_init__(self) -> None:
        mask = _readonly_array(
            self.psd_valid_mask, dtype=bool, label="psd_valid_mask", ndim=1
        )
        ctot = _ctot_quantity(self.ctot, u.mas**2, "ctot", len(mask), dtype=None)
        validate_psf_dtype(ctot.dtype)
        object.__setattr__(self, "ctot", ctot)
        object.__setattr__(self, "psd_valid_mask", mask)
        if self.psd_clipped is not None:
            clipped = _readonly_array(
                self.psd_clipped, dtype=bool, label="psd_clipped", ndim=1
            )
            if clipped.shape != mask.shape:
                raise ValueError("psd_clipped must match the covariance field length.")
            object.__setattr__(self, "psd_clipped", clipped)

    @property
    def psd_valid_count(self) -> int:
        """Number of positions with nonzero PSD covariance."""
        return int(np.count_nonzero(self.psd_valid_mask))

    @property
    def psd_valid_fraction(self) -> float:
        """Fraction of positions with nonzero PSD covariance."""
        return self.psd_valid_count / self.psd_valid_mask.size


@dataclass(frozen=True)
class MastselDiagnostics(_PhysicalRecord):
    """Effective NGS sensing inputs from the MASTSEL calculation.

    Attributes:
        ngs_metrics: Interpolated NGS high-order metrics.
        ngs_flux: Effective photon rates in photon/s, in NGS order.
        frame_rate: Shared low-order WFS sampling rate in Hz.
        angle_to_wavefront_scale: MASTSEL's wavefront conversion in nm/mas.
        effective_ini: Effective MASTSEL configuration at Debug level.
    """

    ngs_metrics: PsfMetricPrediction
    ngs_flux: u.Quantity
    frame_rate: u.Quantity
    angle_to_wavefront_scale: u.Quantity
    effective_ini: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.ngs_metrics, PsfMetricPrediction):
            raise TypeError("ngs_metrics must be a PsfMetricPrediction.")
        dtype = validate_psf_dtype(self.ngs_metrics.dtype)
        flux = _vector_with_length(
            self.ngs_flux, u.photon / u.s, "ngs_flux", self.ngs_metrics.ee.size,
            positive=True, dtype=dtype,
        )
        frame_rate = _positive_scalar(self.frame_rate, u.Hz, "frame_rate", dtype=dtype)
        scale = _positive_scalar(
            self.angle_to_wavefront_scale, u.nm / u.mas, "angle_to_wavefront_scale",
            dtype=dtype,
        )
        if self.effective_ini is not None and (
            not isinstance(self.effective_ini, str) or not self.effective_ini.strip()
        ):
            raise ValueError("effective_ini must be nonempty configuration text.")
        object.__setattr__(self, "ngs_flux", flux)
        object.__setattr__(self, "frame_rate", frame_rate)
        object.__setattr__(self, "angle_to_wavefront_scale", scale)


@dataclass(frozen=True)
class HybridDiagnostics:
    """Validation checks and optional debug outputs for one Hybrid calculation.

    Attributes:
        covariance: Applied angular covariance and PSD checks.
        mastsel: MASTSEL sensing inputs, with the effective INI at Debug level.
        e2e_calibrated: Full E2E-Calibrated prediction at Debug level.
        psfs_ho: Science PSFs before low-order blur at Debug level, in
            ``(N, y, x)`` order.
    """

    covariance: CovarianceDiagnostics
    mastsel: MastselDiagnostics | None = None
    e2e_calibrated: E2ELowOrderPrediction | None = None
    psfs_ho: np.ndarray | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.covariance, CovarianceDiagnostics):
            raise TypeError("covariance must be a CovarianceDiagnostics instance.")
        dtype = self.covariance.ctot.dtype
        field_length = len(self.covariance.ctot)
        if self.mastsel is not None:
            if not isinstance(self.mastsel, MastselDiagnostics):
                raise TypeError("mastsel must be a MastselDiagnostics instance.")
            if self.mastsel.ngs_metrics.dtype != dtype:
                raise ValueError("MASTSEL diagnostics precision must match covariance.")
        if self.e2e_calibrated is not None:
            from ._e2e_prediction import E2ELowOrderPrediction

            if not isinstance(self.e2e_calibrated, E2ELowOrderPrediction):
                raise TypeError("e2e_calibrated must be an E2ELowOrderPrediction.")
            if self.mastsel is not None:
                raise ValueError("A single result can contain only one model's diagnostics.")
            prediction = self.e2e_calibrated
            if prediction.diagnostics is None:
                raise ValueError("E2E-Calibrated debug output requires prediction diagnostics.")
            if (
                prediction.displacement_distribution.dtype != dtype
                or len(prediction.displacement_distribution.covariances) != field_length
            ):
                raise ValueError("The E2E prediction must match covariance precision and length.")
        if self.psfs_ho is not None:
            psfs_ho = _readonly_array(
                self.psfs_ho, dtype=dtype, label="psfs_ho", ndim=3
            )
            if len(psfs_ho) != field_length:
                raise ValueError("psfs_ho must match the covariance field length.")
            object.__setattr__(self, "psfs_ho", psfs_ho)


@dataclass(frozen=True, kw_only=True, eq=False)
class LowOrderDistribution(_PhysicalRecord):
    """Applied centred Gaussian mixture in angular image coordinates.

    Attributes:
        weights: Positive component weights summing to one, (components,).
        covariances: Applied component covariances, (science,components,2,2),
            with physical x/y axes and angular-squared units.
        dtype: Float32 by default or explicit float64 calculation/storage.

    Values have immutable backing and protected unit metadata. Invalid weights,
    dimensions, symmetry or materially non-PSD covariances raise. This record
    describes applied blur without requiring a calibrated provider at load time.
    """

    weights: np.ndarray
    covariances: u.Quantity
    dtype: DTypeLike = np.float32

    def __post_init__(self):
        dtype = validate_psf_dtype(self.dtype)
        weights = _readonly_array(
            self.weights, dtype=dtype, label="mixture weights", ndim=1
        )
        covariance = _readonly_quantity(
            self.covariances, u.mas**2, "mixture covariances", ndim=4, dtype=dtype
        )
        if covariance.shape[1:] != (len(weights), 2, 2):
            raise ValueError(
                "mixture covariances must have shape (science,components,2,2)"
            )
        if np.any(weights <= 0) or not np.isclose(
            weights.sum(), 1, rtol=8 * np.finfo(dtype).eps, atol=0
        ):
            raise ValueError("positive mixture weights must sum to one")
        if not np.allclose(covariance.value, covariance.value.swapaxes(-1, -2)):
            raise ValueError("mixture covariances must be symmetric")
        if np.linalg.eigvalsh(covariance.value).min() < -1e-8:
            raise ValueError("mixture covariances must be positive semidefinite")
        object.__setattr__(self, "dtype", dtype)
        object.__setattr__(self, "weights", weights)
        object.__setattr__(self, "covariances", covariance)

    @property
    def covariance(self) -> u.Quantity:
        """Return the weighted (science,2,2) covariance in mas squared."""
        return np.einsum("c,scab->sab", self.weights, self.covariances.value) * u.mas**2


@dataclass(frozen=True)
class HybridResult(_PhysicalRecord):
    """Final science PSFs and image motion for one Hybrid simulation.

    Attributes:
        positions: Cartesian angular field coordinates in ``(N, 2)`` order.
        psfs: Final HO+LO science PSF cube in ``(N, y, x)`` order.
        jitter: Radial image jitter in mas at each field position.
        metadata: Physical PSF metadata accepted directly by AO Core.
        diagnostics: Optional validation or debug outputs.
    """

    positions: u.Quantity
    psfs: np.ndarray
    jitter: u.Quantity
    metadata: PsfMetadata
    diagnostics: HybridDiagnostics | None = None

    def __post_init__(self) -> None:
        dtype = validate_psf_dtype(np.asarray(self.psfs).dtype)
        psfs = _readonly_array(self.psfs, dtype=dtype, label="psfs", ndim=3)
        if np.any(psfs < 0.0):
            raise ValueError("psfs must be non-negative.")
        if np.any(np.sum(psfs, axis=(-2, -1), dtype=dtype) <= 0.0):
            raise ValueError("psfs must have positive flux.")
        num_science = len(psfs)
        positions = _readonly_quantity(
            self.positions, u.arcsec, "positions", ndim=2, dtype=dtype
        )
        if positions.shape != (num_science, 2):
            raise ValueError(f"positions must have shape ({num_science}, 2).")
        jitter = _vector_with_length(
            self.jitter, u.mas, "jitter", num_science, positive=False, dtype=dtype
        )
        if np.any(jitter.value < 0.0):
            raise ValueError("jitter must be non-negative.")
        metadata = _readonly_metadata(self.metadata, num_psfs=num_science, dtype=dtype)
        diagnostics = self.diagnostics
        if diagnostics is not None:
            if not isinstance(diagnostics, HybridDiagnostics):
                raise TypeError("diagnostics must be a HybridDiagnostics instance.")
            ctot = diagnostics.covariance.ctot
            if ctot.dtype != dtype:
                raise ValueError("diagnostics precision must match the PSFs.")
            if len(ctot) != num_science:
                raise ValueError("diagnostic covariance must match the science field length.")
            if diagnostics.psfs_ho is not None and diagnostics.psfs_ho.shape != psfs.shape:
                raise ValueError("diagnostic psfs_ho must match psfs shape.")
        object.__setattr__(self, "positions", positions)
        object.__setattr__(self, "psfs", psfs)
        object.__setattr__(self, "jitter", jitter)
        object.__setattr__(self, "metadata", metadata)
        object.__setattr__(self, "diagnostics", diagnostics)


def _readonly_array(
    value: object,
    *,
    dtype: np.dtype | type,
    label: str,
    ndim: int,
) -> np.ndarray:
    array = np.array(value, dtype=dtype, copy=True)
    if array.ndim != ndim or any(size == 0 for size in array.shape):
        raise ValueError(f"{label} must be a non-empty {ndim}D array.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{label} must contain only finite values.")
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


def _readonly_quantity(
    value: object,
    unit: u.UnitBase,
    label: str,
    *,
    ndim: int,
    dtype=float,
) -> u.Quantity:
    if not isinstance(value, u.Quantity):
        raise TypeError(f"{label} must be an Astropy Quantity.")
    try:
        array = np.array(
            (value if dtype is None else value.astype(dtype)).to_value(unit),
            dtype=dtype,
            copy=True,
        )
    except u.UnitConversionError as exc:
        raise u.UnitConversionError(f"{label} must be convertible to {unit}.") from exc
    if array.ndim != ndim or any(size == 0 for size in array.shape):
        raise ValueError(f"{label} must be a non-empty {ndim}D quantity.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{label} must contain only finite values.")
    immutable = np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)
    return u.Quantity(immutable, unit, copy=False)


def _scalar(value: object, unit: u.UnitBase, label: str, *, dtype=float) -> u.Quantity:
    quantity = _readonly_quantity(value, unit, label, ndim=0, dtype=dtype)
    return quantity


def _positive_scalar(
    value: object, unit: u.UnitBase, label: str, *, dtype=float
) -> u.Quantity:
    quantity = _scalar(value, unit, label, dtype=dtype)
    if quantity.value <= 0.0:
        raise ValueError(f"{label} must be positive.")
    return quantity


def _vector_with_length(
    value: object,
    unit: u.UnitBase,
    label: str,
    length: int,
    *,
    positive: bool,
    dtype=float,
) -> u.Quantity:
    quantity = _readonly_quantity(value, unit, label, ndim=1, dtype=dtype)
    if quantity.shape != (length,):
        raise ValueError(f"{label} must have shape ({length},), got {quantity.shape}.")
    if positive and np.any(quantity.value <= 0.0):
        raise ValueError(f"{label} must be positive.")
    return quantity


def _ctot_quantity(
    value: object,
    unit: u.UnitBase,
    label: str,
    num_science: int,
    *,
    dtype=float,
) -> u.Quantity:
    quantity = _readonly_quantity(value, unit, label, ndim=3, dtype=dtype)
    expected = (num_science, 2, 2)
    if quantity.shape != expected:
        raise ValueError(f"{label} must have shape {expected}, got {quantity.shape}.")
    if not np.allclose(quantity.value, np.swapaxes(quantity.value, -1, -2)):
        raise ValueError(f"{label} matrices must be symmetric.")
    return quantity


def _jitter_from_ctot(ctot_angle: u.Quantity) -> u.Quantity:
    values = ctot_angle.to_value(u.mas**2)
    return np.sqrt(np.clip(np.trace(values, axis1=-2, axis2=-1), 0.0, None)) * u.mas


def _psd_valid_mask(
    ctot_wavefront: u.Quantity, *, unit=u.nm**2, tolerance=CTOT_PSD_TOLERANCE_NM2
) -> np.ndarray:
    """Return the canonical nonzero-PSD mask using the fixed engine tolerance."""
    values = ctot_wavefront.to_value(unit)
    symmetric = 0.5 * (values + np.swapaxes(values, -1, -2))
    eigenvalues = np.linalg.eigvalsh(symmetric)
    return np.all(eigenvalues >= -tolerance, axis=1) & np.any(
        eigenvalues > 1.0e-12, axis=1
    )


def _displacement_psd_mask(
    ctot: u.Quantity, ctot_wavefront: u.Quantity | None = None
) -> np.ndarray:
    """Check MASTSEL in native units and calibrated covariance in angular units."""
    if ctot_wavefront is not None:
        return _psd_valid_mask(ctot_wavefront)
    return _psd_valid_mask(ctot, unit=u.mas**2, tolerance=1e-8)


def _readonly_metadata(
    metadata: PsfMetadata, *, num_psfs: int, dtype=float
) -> PsfMetadata:
    if not isinstance(metadata, PsfMetadata):
        raise TypeError("metadata must be an ao_core.stats.PsfMetadata instance.")
    wavelength = _shared_or_vector(
        metadata.wavelength, u.um, "metadata.wavelength", num_psfs, dtype=dtype
    )
    pixel_scale = _shared_or_vector(
        metadata.pixel_scale, u.mas, "metadata.pixel_scale", num_psfs, dtype=dtype
    )
    tel_diameter = _positive_scalar(
        metadata.tel_diameter, u.m, "metadata.tel_diameter", dtype=dtype
    )
    tel_pupil = _readonly_quantity(
        metadata.tel_pupil,
        u.dimensionless_unscaled,
        "metadata.tel_pupil",
        ndim=2,
        dtype=dtype,
    )
    if np.any(tel_pupil.value < 0.0):
        raise ValueError("metadata.tel_pupil must be non-negative.")
    if not np.any(tel_pupil.value > 0.0):
        raise ValueError("metadata.tel_pupil must contain positive support.")
    return PsfMetadata(
        wavelength=wavelength,
        pixel_scale=pixel_scale,
        tel_diameter=tel_diameter,
        tel_pupil=tel_pupil,
    )


def _shared_or_vector(
    value: object,
    unit: u.UnitBase,
    label: str,
    length: int,
    *,
    dtype=float,
) -> u.Quantity:
    if not isinstance(value, u.Quantity):
        raise TypeError(f"{label} must be an Astropy Quantity.")
    ndim = np.asarray(value.value).ndim
    if ndim not in (0, 1):
        raise ValueError(f"{label} must be scalar or one-dimensional.")
    quantity = _readonly_quantity(value, unit, label, ndim=ndim, dtype=dtype)
    if ndim == 1 and quantity.shape != (length,):
        raise ValueError(f"{label} must have shape ({length},), got {quantity.shape}.")
    if np.any(quantity.value <= 0.0):
        raise ValueError(f"{label} must be positive.")
    return quantity


__all__ = [
    "CovarianceDiagnostics",
    "DiagnosticsLevel",
    "HybridCtotContext",
    "HybridCtotCorrection",
    "HybridDiagnostics",
    "HybridMastselStatistics",
    "HybridRequest",
    "HybridResult",
    "LowOrderDistribution",
    "LowOrderModel",
    "MastselDiagnostics",
    "NonPsdPolicy",
    "PsfMetricPrediction",
    "PsfMetricProvider",
    "PsfPrediction",
    "PsfProvider",
]
