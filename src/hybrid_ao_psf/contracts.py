"""Immutable public contracts for one Hybrid AO PSF simulation."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np
from ao_stats import PsfMetadata
from astropy import units as u
from numpy.typing import DTypeLike

from ._validation import validate_psf_dtype

if TYPE_CHECKING:
    from mastsel import MavisLO

# Calibrated against released MASTSEL output. Negative eigenvalues at or below
# this magnitude represent a material covariance failure in nm^2.
CTOT_PSD_TOLERANCE_NM2 = 1.0e-9


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
class ScienceHoPsfPrediction:
    """Science high-order PSFs returned by a loaded provider.

    Attributes:
        psfs: One non-negative PSF image per requested science coordinate, in
            ``(N, y, x)`` order. Columns increase toward ``+x`` and rows
            toward ``+y`` in the request's Cartesian frame.
        pixel_scale: Scalar angular pixel scale shared by the returned PSFs.
        metadata: Physical PSF metadata compatible with AO Stats.
        meta: Optional scalar physical metadata propagated from the source.
        dtype: PSF pixel precision, float32 by default. Explicit float64 retains
            double-precision source pixels; other precisions are rejected.
    """

    psfs: np.ndarray
    pixel_scale: u.Quantity
    metadata: PsfMetadata
    meta: Mapping[str, u.Quantity] = field(default_factory=dict)
    dtype: DTypeLike = field(default=np.float32, kw_only=True)

    def __post_init__(self) -> None:
        dtype = validate_psf_dtype(self.dtype)
        psfs = _readonly_array(self.psfs, dtype=dtype, label="psfs", ndim=3)
        object.__setattr__(self, "dtype", dtype)
        if np.any(psfs < 0.0):
            raise ValueError("psfs must be non-negative.")
        flux = np.sum(psfs, axis=(-2, -1), dtype=np.float64)
        if np.any(flux <= 0.0):
            raise ValueError("psfs must have positive flux.")
        pixel_scale = _positive_scalar(self.pixel_scale, u.mas, "pixel_scale")
        metadata = _readonly_metadata(self.metadata, num_psfs=psfs.shape[0])
        if metadata.pixel_scale.ndim != 0 or not np.isclose(
            metadata.pixel_scale.to_value(u.mas), pixel_scale.to_value(u.mas)
        ):
            raise ValueError("pixel_scale must match scalar metadata.pixel_scale.")
        meta = _quantity_mapping(self.meta, label="meta", scalar_only=True)
        object.__setattr__(self, "psfs", psfs)
        object.__setattr__(self, "pixel_scale", pixel_scale)
        object.__setattr__(self, "metadata", metadata)
        object.__setattr__(self, "meta", meta)


@dataclass(frozen=True)
class NgsHoMetricPrediction:
    """NGS high-order metrics returned by a loaded provider.

    Attributes:
        ee: Encircled-energy fraction for each requested NGS coordinate.
        fwhm: Full width at half maximum for each requested NGS coordinate.
        sr: Strehl ratio for each requested NGS coordinate.
    """

    ee: u.Quantity
    fwhm: u.Quantity
    sr: u.Quantity

    def __post_init__(self) -> None:
        ee = _readonly_quantity(self.ee, u.dimensionless_unscaled, "ee", ndim=1)
        fwhm = _readonly_quantity(self.fwhm, u.mas, "fwhm", ndim=1)
        sr = _readonly_quantity(self.sr, u.dimensionless_unscaled, "sr", ndim=1)
        if not (ee.shape == fwhm.shape == sr.shape):
            raise ValueError("NGS metric vectors must have identical shapes.")
        if ee.size == 0:
            raise ValueError("NGS metric vectors must not be empty.")
        if np.any(ee.value < 0.0) or np.any(sr.value < 0.0):
            raise ValueError("ee and sr must be non-negative.")
        if np.any(fwhm.value <= 0.0):
            raise ValueError("fwhm must be positive.")
        object.__setattr__(self, "ee", ee)
        object.__setattr__(self, "fwhm", fwhm)
        object.__setattr__(self, "sr", sr)


@runtime_checkable
class SciencePsfProvider(Protocol):
    """Loaded, reusable provider of interpolated science high-order PSFs."""

    def predict(
        self,
        *,
        zenith_angle: u.Quantity,
        wavelength: u.Quantity,
        x: u.Quantity,
        y: u.Quantity,
    ) -> ScienceHoPsfPrediction:
        """Evaluate science high-order PSFs at one request's coordinates."""
        ...


@runtime_checkable
class NgsMetricProvider(Protocol):
    """Loaded, reusable provider of interpolated NGS high-order metrics."""

    def predict(
        self,
        *,
        zenith_angle: u.Quantity,
        x: u.Quantity,
        y: u.Quantity,
    ) -> NgsHoMetricPrediction:
        """Evaluate NGS high-order metrics at one request's coordinates."""
        ...


@dataclass(frozen=True, kw_only=True)
class HybridRequest:
    """Inputs for one stateless Hybrid AO PSF simulation.

    Attributes:
        science_x: Science field x-coordinates in arcseconds.
        science_y: Science field y-coordinates in arcseconds.
        ngs_x: Natural-guide-star x-coordinates in arcseconds.
        ngs_y: Natural-guide-star y-coordinates in arcseconds. Each x/y pair
            is passed to MASTSEL in that order.
        ngs_flux: Effective photon rate for each active guide star, as an
            alternative to ``ngs_magnitude`` and its zeropoint.
        ngs_magnitude: Magnitude of each active guide star, when supplying
            magnitudes instead of effective flux.
        ngs_magnitude_zeropoint: Caller-supplied pre-aperture zero-magnitude
            photon flux, required only with magnitudes.
        wavelength: Science wavelength.
        zenith_angle: Observation zenith angle.
        mastsel_ini: Complete MASTSEL configuration text.
        non_psd_policy: Handling of materially non-PSD covariance matrices.
        diagnostics_level: Amount of optional intermediate data to retain.
    """

    science_x: u.Quantity
    science_y: u.Quantity
    ngs_x: u.Quantity
    ngs_y: u.Quantity
    ngs_flux: u.Quantity | None = None
    ngs_magnitude: u.Quantity | None = None
    ngs_magnitude_zeropoint: u.Quantity | None = None
    wavelength: u.Quantity
    zenith_angle: u.Quantity
    mastsel_ini: str
    non_psd_policy: NonPsdPolicy | str = NonPsdPolicy.ERROR
    diagnostics_level: DiagnosticsLevel | str = DiagnosticsLevel.NONE

    def __post_init__(self) -> None:
        science_x = _readonly_quantity(self.science_x, u.arcsec, "science_x", ndim=1)
        science_y = _readonly_quantity(self.science_y, u.arcsec, "science_y", ndim=1)
        ngs_x = _readonly_quantity(self.ngs_x, u.arcsec, "ngs_x", ndim=1)
        ngs_y = _readonly_quantity(self.ngs_y, u.arcsec, "ngs_y", ndim=1)
        if (self.ngs_flux is None) == (self.ngs_magnitude is None):
            raise ValueError("Supply exactly one of ngs_flux or ngs_magnitude.")
        ngs_flux = None
        ngs_magnitude = None
        zeropoint = None
        if self.ngs_flux is not None:
            if self.ngs_magnitude_zeropoint is not None:
                raise ValueError("ngs_magnitude_zeropoint requires ngs_magnitude.")
            ngs_flux = _readonly_quantity(
                self.ngs_flux, u.photon / u.s, "ngs_flux", ndim=1
            )
            if np.any(ngs_flux.value <= 0.0):
                raise ValueError("ngs_flux must be positive.")
        else:
            ngs_magnitude = _readonly_quantity(
                self.ngs_magnitude, u.mag, "ngs_magnitude", ndim=1
            )
            zeropoint = _positive_scalar(
                self.ngs_magnitude_zeropoint,
                u.photon / (u.m**2 * u.s),
                "ngs_magnitude_zeropoint",
            )
        if science_x.size == 0:
            raise ValueError("At least one science coordinate is required.")
        if ngs_x.size == 0:
            raise ValueError("At least one active NGS is required.")
        if science_x.shape != science_y.shape:
            raise ValueError("science_x and science_y must have identical shapes.")
        ngs_input = ngs_flux if ngs_flux is not None else ngs_magnitude
        if not (ngs_x.shape == ngs_y.shape == ngs_input.shape):
            raise ValueError(
                "NGS coordinates and flux or magnitude vectors must have identical shapes."
            )
        wavelength = _positive_scalar(self.wavelength, u.um, "wavelength")
        zenith_angle = _scalar(self.zenith_angle, u.deg, "zenith_angle")
        if not 0.0 <= zenith_angle.to_value(u.deg) < 90.0:
            raise ValueError("zenith_angle must be in [0, 90) deg.")
        mastsel_ini = str(self.mastsel_ini).strip()
        if not mastsel_ini:
            raise ValueError("mastsel_ini must not be empty.")
        try:
            non_psd_policy = NonPsdPolicy(self.non_psd_policy)
        except ValueError as exc:
            raise ValueError("non_psd_policy must be 'error' or 'clip'.") from exc
        try:
            diagnostics_level = DiagnosticsLevel(self.diagnostics_level)
        except ValueError as exc:
            raise ValueError(
                "diagnostics_level must be 'none', 'validation', or 'debug'."
            ) from exc
        object.__setattr__(self, "science_x", science_x)
        object.__setattr__(self, "science_y", science_y)
        object.__setattr__(self, "ngs_x", ngs_x)
        object.__setattr__(self, "ngs_y", ngs_y)
        object.__setattr__(self, "ngs_flux", ngs_flux)
        object.__setattr__(self, "ngs_magnitude", ngs_magnitude)
        object.__setattr__(self, "ngs_magnitude_zeropoint", zeropoint)
        object.__setattr__(self, "wavelength", wavelength)
        object.__setattr__(self, "zenith_angle", zenith_angle)
        object.__setattr__(self, "mastsel_ini", mastsel_ini + "\n")
        object.__setattr__(self, "non_psd_policy", non_psd_policy)
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
    """Borrowed runtime context valid only during a Ctot correction call.

    Do not retain, serialize or mutate the runtime or its resolved inputs.
    Hybrid AO PSF owns MASTSEL internals; downstream code should consume the
    mathematical statistics rather than re-evaluating the baseline.

    Attributes:
        mastsel: Live MavisLO instance that calculated baseline Ctot.
        mastsel_statistics: Captured baseline statistics; absent during private
            preparation or when no correction definition is supplied.
        angle_to_wavefront_scale: Positive scalar conversion in nm/mas.
        request: Immutable request with active NGS and science coordinates.
        ngs_metrics: The interpolated metrics supplied to baseline MASTSEL.
        ngs_flux: Effective per-NGS photon rates, in photon/s.
        ngs_frame_rate: Resolved per-NGS frame rates, in Hz.
        effective_mastsel_ini: Complete configuration used by the live runtime.
    """

    mastsel: MavisLO
    mastsel_statistics: HybridMastselStatistics | None
    angle_to_wavefront_scale: u.Quantity
    request: HybridRequest
    ngs_metrics: NgsHoMetricPrediction
    ngs_flux: u.Quantity
    ngs_frame_rate: u.Quantity
    effective_mastsel_ini: str

    def __post_init__(self) -> None:
        if not isinstance(self.request, HybridRequest):
            raise TypeError("request must be a HybridRequest.")
        if not isinstance(self.ngs_metrics, NgsHoMetricPrediction):
            raise TypeError("ngs_metrics must be an NgsHoMetricPrediction.")
        size = self.request.ngs_x.size
        if self.ngs_metrics.ee.size != size:
            raise ValueError("NGS metrics must match the request's active NGS.")
        object.__setattr__(
            self,
            "angle_to_wavefront_scale",
            _positive_scalar(
                self.angle_to_wavefront_scale, u.nm / u.mas, "angle_to_wavefront_scale"
            ),
        )
        for name, unit in (("ngs_flux", u.photon / u.s), ("ngs_frame_rate", u.Hz)):
            object.__setattr__(
                self,
                name,
                _vector_with_length(
                    getattr(self, name), unit, name, size, positive=True
                ),
            )
        if (
            not isinstance(self.effective_mastsel_ini, str)
            or not self.effective_mastsel_ini.strip()
        ):
            raise ValueError(
                "effective_mastsel_ini must be nonempty configuration text."
            )
        statistics = self.mastsel_statistics
        if statistics is not None:
            if not isinstance(statistics, HybridMastselStatistics):
                raise TypeError(
                    "mastsel_statistics must be HybridMastselStatistics or None."
                )
            if statistics.science_ngs_covariance.shape != (
                self.request.science_x.size,
                2,
                2 * size,
            ):
                raise ValueError(
                    "MASTSEL statistics must match the request's field and NGS."
                )


@dataclass(frozen=True)
class HybridCtotCorrection:
    """A field-wide covariance callback paired with its static provenance.

    Attributes:
        apply: Callable ``apply(ctot, context)`` receiving unconditioned
            ``(Nscience, 2, 2)`` baseline Ctot in nm² and a borrowed live context.
            Return a replacement Quantity convertible to nm² with identical
            shape/order, or None for no change. Do not mutate borrowed inputs.
            Exceptions propagate; conditioning and the single blur stay in
            the engine.
        provenance: String-to-string mapping copied and frozen at construction.
            Describes configured identity even if apply returns None; keys must
            not collide with engine provenance. Not computed per science point.
    """

    apply: Callable[[u.Quantity, HybridCtotContext], u.Quantity | None]
    provenance: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not callable(self.apply):
            raise TypeError("apply must be callable.")
        if not isinstance(self.provenance, Mapping) or any(
            not isinstance(key, str) or not isinstance(value, str)
            for key, value in self.provenance.items()
        ):
            raise TypeError("Correction provenance must be a string-to-string mapping.")
        object.__setattr__(self, "provenance", MappingProxyType(dict(self.provenance)))


@dataclass(frozen=True)
class HybridDiagnostics:
    """Optional scientific validation summaries and debug products.

    Attributes:
        psd_valid_mask: Per-field mask identifying nonzero PSD covariance matrices.
        ctot_angle_trace_min: Minimum angular Ctot trace.
        ctot_angle_trace_max: Maximum angular Ctot trace.
        ctot_angle_trace_mean: Mean angular Ctot trace.
        ctot_angle_determinant_min: Minimum angular Ctot determinant.
        ctot_angle_determinant_max: Maximum angular Ctot determinant.
        ctot_angle_determinant_mean: Mean angular Ctot determinant.
        pre_blur_psfs: Optional science high-order PSFs before low-order blurring.
    """

    psd_valid_mask: np.ndarray
    ctot_angle_trace_min: u.Quantity
    ctot_angle_trace_max: u.Quantity
    ctot_angle_trace_mean: u.Quantity
    ctot_angle_determinant_min: u.Quantity
    ctot_angle_determinant_max: u.Quantity
    ctot_angle_determinant_mean: u.Quantity
    pre_blur_psfs: np.ndarray | None = None

    def __post_init__(self) -> None:
        psd_valid_mask = _readonly_array(
            self.psd_valid_mask,
            dtype=bool,
            label="psd_valid_mask",
            ndim=1,
        )
        trace_min = _scalar(self.ctot_angle_trace_min, u.mas**2, "ctot_angle_trace_min")
        trace_max = _scalar(self.ctot_angle_trace_max, u.mas**2, "ctot_angle_trace_max")
        trace_mean = _scalar(
            self.ctot_angle_trace_mean, u.mas**2, "ctot_angle_trace_mean"
        )
        determinant_min = _scalar(
            self.ctot_angle_determinant_min,
            u.mas**4,
            "ctot_angle_determinant_min",
        )
        determinant_max = _scalar(
            self.ctot_angle_determinant_max,
            u.mas**4,
            "ctot_angle_determinant_max",
        )
        determinant_mean = _scalar(
            self.ctot_angle_determinant_mean,
            u.mas**4,
            "ctot_angle_determinant_mean",
        )
        if not trace_min <= trace_mean <= trace_max:
            raise ValueError("Ctot trace summaries must satisfy min <= mean <= max.")
        if not determinant_min <= determinant_mean <= determinant_max:
            raise ValueError(
                "Ctot determinant summaries must satisfy min <= mean <= max."
            )
        object.__setattr__(self, "psd_valid_mask", psd_valid_mask)
        object.__setattr__(self, "ctot_angle_trace_min", trace_min)
        object.__setattr__(self, "ctot_angle_trace_max", trace_max)
        object.__setattr__(self, "ctot_angle_trace_mean", trace_mean)
        object.__setattr__(self, "ctot_angle_determinant_min", determinant_min)
        object.__setattr__(self, "ctot_angle_determinant_max", determinant_max)
        object.__setattr__(self, "ctot_angle_determinant_mean", determinant_mean)
        if self.pre_blur_psfs is not None:
            object.__setattr__(
                self,
                "pre_blur_psfs",
                _readonly_array(
                    self.pre_blur_psfs,
                    dtype=np.float32,
                    label="pre_blur_psfs",
                    ndim=3,
                ),
            )

    @property
    def psd_valid_count(self) -> int:
        """Number of fields with nonzero positive-semidefinite covariance."""
        return int(np.count_nonzero(self.psd_valid_mask))

    @property
    def psd_valid_fraction(self) -> float:
        """Fraction of fields with nonzero positive-semidefinite covariance."""
        return self.psd_valid_count / int(self.psd_valid_mask.size)


@dataclass(frozen=True)
class HybridResult:
    """Immutable standard result of one Hybrid AO PSF simulation.

    Attributes:
        psfs: Final PSFs after low-order blurring, in ``(N, y, x)`` order.
        metadata: Physical metadata for ``psfs``.
        source_metadata: Scalar metadata from the science PSF provider.
        ngs_metrics: Interpolated guide-star high-order metrics.
        ctot_wavefront: Conditioned low-order covariance in wavefront units,
            with matrix axes ordered ``(x, y)``.
        ctot_angle: The same ``(x, y)`` covariance in angular units.
        angle_to_wavefront_scale: Scale relating angular and wavefront Ctot.
        ngs_flux: Effective guide-star photon rates.
        ngs_magnitude: Optional supplied guide-star magnitudes.
        ngs_magnitude_zeropoint: Optional supplied pre-aperture zeropoint.
        ngs_frame_rate: Effective guide-star frame rates.
        jitter: Radial jitter derived from angular Ctot.
        effective_mastsel_ini: Exact configuration supplied to MASTSEL.
        psd_clipped: Per-field mask recording covariance clipping.
        provenance: String identities for inputs and providers.
        diagnostics: Optional intermediate products.
    """

    psfs: np.ndarray
    metadata: PsfMetadata
    source_metadata: Mapping[str, u.Quantity]
    ngs_metrics: NgsHoMetricPrediction
    ctot_wavefront: u.Quantity
    ctot_angle: u.Quantity
    angle_to_wavefront_scale: u.Quantity
    ngs_flux: u.Quantity
    ngs_frame_rate: u.Quantity
    jitter: u.Quantity
    effective_mastsel_ini: str
    psd_clipped: np.ndarray
    provenance: Mapping[str, str] = field(default_factory=dict)
    diagnostics: HybridDiagnostics | None = None
    ngs_magnitude: u.Quantity | None = None
    ngs_magnitude_zeropoint: u.Quantity | None = None

    def __post_init__(self) -> None:
        psfs = _readonly_array(self.psfs, dtype=np.float32, label="psfs", ndim=3)
        if np.any(psfs < 0.0):
            raise ValueError("psfs must be non-negative.")
        flux = np.sum(psfs, axis=(-2, -1), dtype=np.float64)
        if np.any(flux <= 0.0):
            raise ValueError("psfs must have positive flux.")
        num_science = psfs.shape[0]
        metadata = _readonly_metadata(self.metadata, num_psfs=num_science)
        source_metadata = _quantity_mapping(
            self.source_metadata, label="source_metadata", scalar_only=True
        )
        if not isinstance(self.ngs_metrics, NgsHoMetricPrediction):
            raise TypeError("ngs_metrics must be an NgsHoMetricPrediction.")
        num_ngs = self.ngs_metrics.ee.size
        ctot_wavefront = _ctot_quantity(
            self.ctot_wavefront, u.nm**2, "ctot_wavefront", num_science
        )
        ctot_angle = _ctot_quantity(
            self.ctot_angle, u.mas**2, "ctot_angle", num_science
        )
        angle_to_wavefront_scale = _positive_scalar(
            self.angle_to_wavefront_scale,
            u.nm / u.mas,
            "angle_to_wavefront_scale",
        )
        expected_ctot_angle = (
            ctot_wavefront.to_value(u.nm**2)
            / angle_to_wavefront_scale.to_value(u.nm / u.mas) ** 2
        )
        if not np.allclose(
            ctot_angle.to_value(u.mas**2),
            expected_ctot_angle,
            rtol=1.0e-10,
            atol=1.0e-9,
        ):
            raise ValueError(
                "ctot_angle must equal ctot_wavefront divided by "
                "angle_to_wavefront_scale squared."
            )
        minimum_eigenvalue = float(
            np.min(np.linalg.eigvalsh(ctot_wavefront.to_value(u.nm**2)))
        )
        if minimum_eigenvalue < -CTOT_PSD_TOLERANCE_NM2:
            raise ValueError(
                "ctot_wavefront must be positive semidefinite within the fixed "
                f"{CTOT_PSD_TOLERANCE_NM2:.6g} nm2 tolerance."
            )
        ngs_flux = _vector_with_length(
            self.ngs_flux, u.photon / u.s, "ngs_flux", num_ngs, positive=True
        )
        if (self.ngs_magnitude is None) != (self.ngs_magnitude_zeropoint is None):
            raise ValueError(
                "ngs_magnitude and ngs_magnitude_zeropoint must be paired."
            )
        ngs_magnitude = None
        zeropoint = None
        if self.ngs_magnitude is not None:
            ngs_magnitude = _vector_with_length(
                self.ngs_magnitude, u.mag, "ngs_magnitude", num_ngs, positive=False
            )
            zeropoint = _positive_scalar(
                self.ngs_magnitude_zeropoint,
                u.photon / (u.m**2 * u.s),
                "ngs_magnitude_zeropoint",
            )
        ngs_frame_rate = _vector_with_length(
            self.ngs_frame_rate, u.Hz, "ngs_frame_rate", num_ngs, positive=True
        )
        jitter = _vector_with_length(
            self.jitter, u.mas, "jitter", num_science, positive=False
        )
        if np.any(jitter.value < 0.0):
            raise ValueError("jitter must be non-negative.")
        expected_jitter = _jitter_from_ctot(ctot_angle)
        if not np.allclose(
            jitter.to_value(u.mas),
            expected_jitter.to_value(u.mas),
            rtol=1.0e-6,
            atol=1.0e-6,
        ):
            raise ValueError(
                "jitter must equal the radial jitter derived from ctot_angle."
            )
        psd_clipped = _readonly_array(
            self.psd_clipped, dtype=bool, label="psd_clipped", ndim=1
        )
        if psd_clipped.shape != (num_science,):
            raise ValueError(
                f"psd_clipped must have shape ({num_science},), got {psd_clipped.shape}."
            )
        effective_mastsel_ini = str(self.effective_mastsel_ini).strip()
        if not effective_mastsel_ini:
            raise ValueError("effective_mastsel_ini must not be empty.")
        provenance = MappingProxyType(
            {str(key): str(value) for key, value in self.provenance.items()}
        )
        if self.diagnostics is not None:
            if not isinstance(self.diagnostics, HybridDiagnostics):
                raise TypeError("diagnostics must be a HybridDiagnostics instance.")
            if self.diagnostics.psd_valid_mask.shape != (num_science,):
                raise ValueError(
                    "diagnostic psd_valid_mask must match the science field length."
                )
            expected_psd_valid_mask = _psd_valid_mask(ctot_wavefront)
            if not np.array_equal(
                self.diagnostics.psd_valid_mask, expected_psd_valid_mask
            ):
                raise ValueError(
                    "diagnostic psd_valid_mask must match conditioned ctot_wavefront."
                )
            expected_summaries = _ctot_angle_summaries(ctot_angle)
            for name, expected in expected_summaries.items():
                actual = getattr(self.diagnostics, name)
                if not np.isclose(
                    actual.to_value(expected.unit),
                    expected.value,
                    rtol=1.0e-6,
                    atol=1.0e-9,
                ):
                    raise ValueError(
                        f"diagnostic {name} must match conditioned ctot_angle."
                    )
            pre_blur = self.diagnostics.pre_blur_psfs
            if pre_blur is not None and pre_blur.shape != psfs.shape:
                raise ValueError("diagnostic pre_blur_psfs must match psfs shape.")
        object.__setattr__(self, "psfs", psfs)
        object.__setattr__(self, "metadata", metadata)
        object.__setattr__(self, "source_metadata", source_metadata)
        object.__setattr__(self, "ctot_wavefront", ctot_wavefront)
        object.__setattr__(self, "ctot_angle", ctot_angle)
        object.__setattr__(self, "angle_to_wavefront_scale", angle_to_wavefront_scale)
        object.__setattr__(self, "ngs_flux", ngs_flux)
        object.__setattr__(self, "ngs_magnitude", ngs_magnitude)
        object.__setattr__(self, "ngs_magnitude_zeropoint", zeropoint)
        object.__setattr__(self, "ngs_frame_rate", ngs_frame_rate)
        object.__setattr__(self, "jitter", jitter)
        object.__setattr__(self, "effective_mastsel_ini", effective_mastsel_ini + "\n")
        object.__setattr__(self, "psd_clipped", psd_clipped)
        object.__setattr__(self, "provenance", provenance)


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
    array.flags.writeable = False
    return array


def _readonly_quantity(
    value: object,
    unit: u.UnitBase,
    label: str,
    *,
    ndim: int,
) -> u.Quantity:
    if not isinstance(value, u.Quantity):
        raise TypeError(f"{label} must be an Astropy Quantity.")
    try:
        array = np.array(value.to_value(unit), dtype=float, copy=True)
    except u.UnitConversionError as exc:
        raise u.UnitConversionError(f"{label} must be convertible to {unit}.") from exc
    if array.ndim != ndim or any(size == 0 for size in array.shape):
        raise ValueError(f"{label} must be a non-empty {ndim}D quantity.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{label} must contain only finite values.")
    array.flags.writeable = False
    return u.Quantity(array, unit, copy=False)


def _scalar(value: object, unit: u.UnitBase, label: str) -> u.Quantity:
    quantity = _readonly_quantity(value, unit, label, ndim=0)
    return quantity


def _positive_scalar(value: object, unit: u.UnitBase, label: str) -> u.Quantity:
    quantity = _scalar(value, unit, label)
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
) -> u.Quantity:
    quantity = _readonly_quantity(value, unit, label, ndim=1)
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
) -> u.Quantity:
    quantity = _readonly_quantity(value, unit, label, ndim=3)
    expected = (num_science, 2, 2)
    if quantity.shape != expected:
        raise ValueError(f"{label} must have shape {expected}, got {quantity.shape}.")
    if not np.allclose(quantity.value, np.swapaxes(quantity.value, -1, -2)):
        raise ValueError(f"{label} matrices must be symmetric.")
    return quantity


def _jitter_from_ctot(ctot_angle: u.Quantity) -> u.Quantity:
    values = ctot_angle.to_value(u.mas**2)
    return np.sqrt(np.clip(np.trace(values, axis1=-2, axis2=-1), 0.0, None)) * u.mas


def _psd_valid_mask(ctot_wavefront: u.Quantity) -> np.ndarray:
    """Return the canonical nonzero-PSD mask using the fixed engine tolerance."""
    values = ctot_wavefront.to_value(u.nm**2)
    symmetric = 0.5 * (values + np.swapaxes(values, -1, -2))
    eigenvalues = np.linalg.eigvalsh(symmetric)
    return np.all(eigenvalues >= -CTOT_PSD_TOLERANCE_NM2, axis=1) & np.any(
        eigenvalues > 1.0e-12, axis=1
    )


def _ctot_angle_summaries(ctot_angle: u.Quantity) -> dict[str, u.Quantity]:
    """Return the canonical persisted validation summaries for angular Ctot."""
    values = ctot_angle.to_value(u.mas**2)
    traces = np.trace(values, axis1=-2, axis2=-1)
    determinants = np.linalg.det(values)
    return {
        "ctot_angle_trace_min": np.min(traces) * u.mas**2,
        "ctot_angle_trace_max": np.max(traces) * u.mas**2,
        "ctot_angle_trace_mean": np.mean(traces) * u.mas**2,
        "ctot_angle_determinant_min": np.min(determinants) * u.mas**4,
        "ctot_angle_determinant_max": np.max(determinants) * u.mas**4,
        "ctot_angle_determinant_mean": np.mean(determinants) * u.mas**4,
    }


def _readonly_metadata(metadata: PsfMetadata, *, num_psfs: int) -> PsfMetadata:
    if not isinstance(metadata, PsfMetadata):
        raise TypeError("metadata must be an ao_stats.PsfMetadata instance.")
    wavelength = _shared_or_vector(
        metadata.wavelength, u.um, "metadata.wavelength", num_psfs
    )
    pixel_scale = _shared_or_vector(
        metadata.pixel_scale, u.mas, "metadata.pixel_scale", num_psfs
    )
    tel_diameter = _positive_scalar(metadata.tel_diameter, u.m, "metadata.tel_diameter")
    tel_pupil = _readonly_quantity(
        metadata.tel_pupil,
        u.dimensionless_unscaled,
        "metadata.tel_pupil",
        ndim=2,
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
) -> u.Quantity:
    if not isinstance(value, u.Quantity):
        raise TypeError(f"{label} must be an Astropy Quantity.")
    ndim = np.asarray(value.value).ndim
    if ndim not in (0, 1):
        raise ValueError(f"{label} must be scalar or one-dimensional.")
    quantity = _readonly_quantity(value, unit, label, ndim=ndim)
    if ndim == 1 and quantity.shape != (length,):
        raise ValueError(f"{label} must have shape ({length},), got {quantity.shape}.")
    if np.any(quantity.value <= 0.0):
        raise ValueError(f"{label} must be positive.")
    return quantity


def _quantity_mapping(
    value: Mapping[str, u.Quantity],
    *,
    label: str,
    scalar_only: bool,
) -> Mapping[str, u.Quantity]:
    normalized: dict[str, u.Quantity] = {}
    for raw_name, raw_value in value.items():
        name = str(raw_name).strip()
        if not name or "/" in name:
            raise ValueError(
                f"{label} names must be non-empty and must not contain '/'."
            )
        if not isinstance(raw_value, u.Quantity):
            raise TypeError(f"{label}[{name!r}] must be an Astropy Quantity.")
        array = np.array(raw_value.value, copy=True)
        if scalar_only and array.ndim != 0:
            raise ValueError(f"{label}[{name!r}] must be scalar.")
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{label}[{name!r}] must be finite.")
        array.flags.writeable = False
        normalized[name] = u.Quantity(array, raw_value.unit, copy=False)
    return MappingProxyType(normalized)


__all__ = [
    "DiagnosticsLevel",
    "HybridCtotContext",
    "HybridCtotCorrection",
    "HybridDiagnostics",
    "HybridMastselStatistics",
    "HybridRequest",
    "HybridResult",
    "NgsHoMetricPrediction",
    "NgsMetricProvider",
    "NonPsdPolicy",
    "ScienceHoPsfPrediction",
    "SciencePsfProvider",
]
