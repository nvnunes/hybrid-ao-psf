"""Stateless MASTSEL-backed Hybrid AO PSF simulation engine."""

from __future__ import annotations

import tempfile
from configparser import ConfigParser
from io import StringIO

import numpy as np
from astropy import units as u
from ngs_photometry import (
    WFSPhotometryConfig,
    magnitudes_to_effective_photons_per_second,
)
from scipy.fft import fft2, ifft2

from ._mastsel import (
    _capture_mastsel,
    _complete_mastsel_context,
    _compute_mastsel_ctot,
    _prepare_mastsel_context,
)
from ._units import quantity_value
from ._validation import validate_psf_dtype
from .contracts import (
    CTOT_PSD_TOLERANCE_NM2,
    DiagnosticsLevel,
    HybridCtotCorrection,
    HybridDiagnostics,
    HybridRequest,
    HybridResult,
    NgsMetricProvider,
    NonPsdPolicy,
    SciencePsfProvider,
    _ctot_angle_summaries,
    _jitter_from_ctot,
    _psd_valid_mask,
)


def simulate(
    request: HybridRequest,
    science_provider: SciencePsfProvider,
    ngs_provider: NgsMetricProvider,
    *,
    ctot_correction: HybridCtotCorrection | None = None,
) -> HybridResult:
    """Run one Hybrid AO PSF simulation.

    The request supplies effective NGS flux or magnitudes with a caller-owned
    zeropoint. Magnitudes are converted using MASTSEL telescope/WFS geometry
    before Ctot evaluation. Providers are borrowed and reusable. The result
    owns its arrays, and the engine retains no mutable per-run state.

    An optional correction receives the unconditioned whole-field covariance
    and a borrowed live runtime context. It returns a replacement in compatible
    wavefront covariance units, or None. The engine conditions and blurs once;
    exceptions propagate. Static correction provenance is merged with provider
    identities; collisions are errors. The context is not retained or persisted.
    """
    if not isinstance(request, HybridRequest):
        raise TypeError("request must be a HybridRequest.")
    if not isinstance(science_provider, SciencePsfProvider):
        raise TypeError("science_provider must implement SciencePsfProvider.")
    if not isinstance(ngs_provider, NgsMetricProvider):
        raise TypeError("ngs_provider must implement NgsMetricProvider.")
    if ctot_correction is not None and not isinstance(
        ctot_correction, HybridCtotCorrection
    ):
        raise TypeError("ctot_correction must be HybridCtotCorrection or None.")
    provenance = {
        "science_provider": _type_identity(science_provider),
        "ngs_provider": _type_identity(ngs_provider),
    }
    if ctot_correction is not None:
        collisions = provenance.keys() & ctot_correction.provenance.keys()
        if collisions:
            raise ValueError(
                f"Correction provenance collides with engine keys: {sorted(collisions)}."
            )
        provenance.update(ctot_correction.provenance)

    # Concrete fitted providers prepare once; protocol-only providers have their
    # returned physical records converted at the same operation boundary.
    if (
        hasattr(science_provider, "prepare")
        and getattr(science_provider, "dtype", None) != request.dtype
    ):
        science_provider = science_provider.prepare(dtype=request.dtype)
    if (
        hasattr(ngs_provider, "prepare")
        and getattr(ngs_provider, "dtype", None) != request.dtype
    ):
        ngs_provider = ngs_provider.prepare(dtype=request.dtype)
    science = science_provider.predict(
        zenith_angle=request.zenith_angle,
        wavelength=request.wavelength,
        x=request.science_x,
        y=request.science_y,
    )
    if science.psfs.shape[0] != request.science_x.size:
        raise ValueError("Science provider output length does not match the request.")
    metrics = ngs_provider.predict(
        zenith_angle=request.zenith_angle,
        x=request.ngs_x,
        y=request.ngs_y,
    )
    if metrics.ee.size != request.ngs_x.size:
        raise ValueError("NGS provider output length does not match the request.")

    from dataclasses import replace

    science = replace(science, dtype=request.dtype)
    metrics = replace(metrics, dtype=request.dtype)
    effective_ini, ngs_frame_rate, ngs_flux = _resolve_effective_mastsel(request)
    with tempfile.TemporaryDirectory(prefix="hybrid_ao_psf_") as temporary:
        context = _prepare_mastsel_context(
            temporary, request, metrics, effective_ini, ngs_frame_rate, ngs_flux
        )
        if ctot_correction is None:
            ctot_raw = _compute_mastsel_ctot(context)
        else:
            with _capture_mastsel(context) as captured:
                ctot_raw = _compute_mastsel_ctot(context)
            context = _complete_mastsel_context(context, captured)
        validate_ctot_shape(
            ctot_raw, expected_size=request.science_x.size, label="MASTSEL Ctot"
        )
        angle_to_wavefront_scale = context.angle_to_wavefront_scale
        ctot_raw = u.Quantity(ctot_raw, dtype=request.dtype)
        if ctot_correction is not None:
            ctot_raw.flags.writeable = False
            corrected = ctot_correction.apply(ctot_raw, context)
            if corrected is not None:
                validate_ctot_shape(
                    corrected,
                    expected_size=request.science_x.size,
                    label="Corrected Ctot",
                )
                ctot_raw = corrected.to(u.nm**2)
    ctot_wavefront, psd_clipped = condition_ctot(
        ctot_raw,
        policy=request.non_psd_policy,
        dtype=request.dtype,
    )
    angle_to_wavefront_scale = u.Quantity(angle_to_wavefront_scale, dtype=request.dtype)
    ctot_angle = ctot_wavefront / angle_to_wavefront_scale**2
    pre_blur = np.asarray(science.psfs, dtype=request.dtype).copy()
    psfs = pre_blur.copy()
    apply_ctot_blur(
        psfs,
        ctot_wavefront,
        pixel_scale=science.pixel_scale,
        angle_to_wavefront_scale=angle_to_wavefront_scale,
    )
    diagnostics = None
    if request.diagnostics_level is not DiagnosticsLevel.NONE:
        diagnostics = _build_diagnostics(
            ctot_wavefront,
            ctot_angle,
            pre_blur_psfs=(
                pre_blur
                if request.diagnostics_level is DiagnosticsLevel.DEBUG
                else None
            ),
        )
    return HybridResult(
        dtype=request.dtype,
        psfs=psfs,
        metadata=science.metadata,
        source_metadata=science.meta,
        ngs_metrics=metrics,
        ctot_wavefront=ctot_wavefront,
        ctot_angle=ctot_angle,
        angle_to_wavefront_scale=angle_to_wavefront_scale,
        ngs_flux=ngs_flux,
        ngs_magnitude=request.ngs_magnitude,
        ngs_magnitude_zeropoint=request.ngs_magnitude_zeropoint,
        ngs_frame_rate=ngs_frame_rate,
        jitter=_jitter_from_ctot(ctot_angle),
        effective_mastsel_ini=effective_ini,
        psd_clipped=psd_clipped,
        provenance=provenance,
        diagnostics=diagnostics,
    )


def _build_diagnostics(
    ctot_wavefront: u.Quantity,
    ctot_angle: u.Quantity,
    *,
    pre_blur_psfs: np.ndarray | None,
) -> HybridDiagnostics:
    return HybridDiagnostics(
        dtype=ctot_wavefront.dtype,
        psd_valid_mask=psd_valid_mask(ctot_wavefront),
        **_ctot_angle_summaries(ctot_angle),
        pre_blur_psfs=pre_blur_psfs,
    )


def _type_identity(value: object) -> str:
    value_type = type(value)
    return f"{value_type.__module__}:{value_type.__qualname__}"


def condition_ctot(
    ctot_wavefront: u.Quantity,
    *,
    policy: NonPsdPolicy | str = NonPsdPolicy.ERROR,
    dtype=np.float32,
) -> tuple[u.Quantity, np.ndarray]:
    """Symmetrize Ctot and apply the fixed numerical/non-PSD policy."""
    dtype = validate_psf_dtype(dtype)
    validate_ctot_shape(ctot_wavefront, label="Ctot")
    try:
        policy = NonPsdPolicy(policy)
    except ValueError as exc:
        raise ValueError("policy must be 'error' or 'clip'.") from exc
    values = quantity_value(
        ctot_wavefront, u.nm**2, label="ctot_wavefront", dtype=dtype
    )
    symmetric = 0.5 * (values + np.swapaxes(values, -1, -2))
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    materially_invalid = np.any(eigenvalues < -CTOT_PSD_TOLERANCE_NM2, axis=1)
    if policy is NonPsdPolicy.ERROR and np.any(materially_invalid):
        minimum = float(np.min(eigenvalues))
        raise ValueError(
            "Ctot is materially non-positive-semidefinite: "
            f"minimum eigenvalue {minimum:.6g} nm2 is below "
            f"-{CTOT_PSD_TOLERANCE_NM2:.6g} nm2."
        )
    clipped = np.any(eigenvalues < 0.0, axis=1)
    if np.any(clipped):
        clipped_eigenvalues = np.maximum(eigenvalues[clipped], 0.0)
        symmetric[clipped] = np.einsum(
            "...ik,...k,...jk->...ij",
            eigenvectors[clipped],
            clipped_eigenvalues,
            eigenvectors[clipped],
        )
    return symmetric * u.nm**2, clipped


def _resolve_effective_mastsel(
    request: HybridRequest,
) -> tuple[str, u.Quantity, u.Quantity]:
    """Resolve the effective MASTSEL INI, frame rates, and effective flux."""
    parser = ConfigParser(interpolation=None)
    parser.optionxform = str
    parser.read_string(request.mastsel_ini)
    for section in ("telescope", "sources_LO"):
        if not parser.has_section(section):
            raise ValueError(f"MASTSEL configuration requires [{section}].")
    if not parser.has_section("RTC") or "SensorFrameRate_LO" not in parser["RTC"]:
        raise ValueError("MASTSEL configuration requires RTC.SensorFrameRate_LO.")
    try:
        frame_rate = float(parser["RTC"]["SensorFrameRate_LO"])
    except ValueError as exc:
        raise ValueError(
            "RTC.SensorFrameRate_LO must be a positive finite scalar."
        ) from exc
    if not np.isfinite(frame_rate) or frame_rate <= 0.0:
        raise ValueError("RTC.SensorFrameRate_LO must be a positive finite scalar.")
    if not parser.has_section("sources_science"):
        parser.add_section("sources_science")
    parser["telescope"]["ZenithAngle"] = _format_scalar(
        request.zenith_angle.to_value(u.deg)
    )
    science_r, science_theta = _cartesian_to_polar(request.science_x, request.science_y)
    ngs_r, ngs_theta = _cartesian_to_polar(request.ngs_x, request.ngs_y)
    parser["sources_science"]["Wavelength"] = _format_array(
        np.asarray([request.wavelength.to_value(u.m)])
    )
    parser["sources_science"]["Zenith"] = _format_array(science_r.to_value(u.arcsec))
    parser["sources_science"]["Azimuth"] = _format_array(science_theta.to_value(u.deg))
    parser["sources_LO"]["Zenith"] = _format_array(ngs_r.to_value(u.arcsec))
    parser["sources_LO"]["Azimuth"] = _format_array(ngs_theta.to_value(u.deg))
    if parser.has_section("sensor_LO") and "NumberLenslets" in parser["sensor_LO"]:
        lenslets = _parse_numeric_array(parser["sensor_LO"]["NumberLenslets"])
        if lenslets.size == 0:
            raise ValueError("sensor_LO.NumberLenslets must not be empty.")
        parser["sensor_LO"]["NumberLenslets"] = _format_array(
            np.full(request.ngs_x.size, lenslets[0], dtype=float)
        )
    ngs_frame_rate = np.full(request.ngs_x.size, frame_rate, dtype=float) * u.Hz
    if request.ngs_flux is not None:
        ngs_flux = request.ngs_flux
    else:
        if "TelescopeDiameter" not in parser["telescope"]:
            raise ValueError(
                "MASTSEL configuration requires telescope.TelescopeDiameter for magnitudes."
            )
        if (
            not parser.has_section("sensor_LO")
            or "NumberLenslets" not in parser["sensor_LO"]
        ):
            raise ValueError(
                "MASTSEL configuration requires sensor_LO.NumberLenslets for magnitudes."
            )
        try:
            diameter = float(parser["telescope"]["TelescopeDiameter"])
        except ValueError as exc:
            raise ValueError("telescope.TelescopeDiameter must be numeric.") from exc
        lenslets = _parse_numeric_array(parser["sensor_LO"]["NumberLenslets"])
        if lenslets.size == 0:
            raise ValueError("sensor_LO.NumberLenslets must not be empty.")
        photometry = WFSPhotometryConfig(
            telescope_diameter=diameter * u.m,
            n_channels=float(lenslets[0]),
            frame_rate=frame_rate * u.Hz,
            zeropoint=request.ngs_magnitude_zeropoint,
        )
        ngs_flux = magnitudes_to_effective_photons_per_second(
            request.ngs_magnitude, photometry
        )
        if np.any(ngs_flux.to_value(u.photon / u.s) <= 0.0):
            raise ValueError("ngs_magnitude resolves to non-positive ngs_flux.")
    stream = StringIO()
    parser.write(stream)
    return stream.getvalue(), ngs_frame_rate, ngs_flux


def _cartesian_to_polar(x: u.Quantity, y: u.Quantity) -> tuple[u.Quantity, u.Quantity]:
    x_value = x.to_value(u.arcsec)
    y_value = y.to_value(u.arcsec)
    radius = np.hypot(x_value, y_value)
    theta = np.mod(np.rad2deg(np.arctan2(y_value, x_value)), 360.0)
    theta[radius == 0.0] = 0.0
    return radius * u.arcsec, theta * u.deg


def _format_scalar(value: float) -> str:
    return f"{float(value):.16g}"


def _format_array(values: np.ndarray) -> str:
    return "[" + ", ".join(_format_scalar(value) for value in values) + "]"


def _parse_numeric_array(value: str) -> np.ndarray:
    stripped = str(value).strip().strip("[]")
    if not stripped:
        return np.array([], dtype=float)
    try:
        return np.asarray([float(item) for item in stripped.split(",")], dtype=float)
    except ValueError as exc:
        raise ValueError("MASTSEL numeric array contains a non-numeric value.") from exc


def apply_ctot_blur(
    psfs: np.ndarray,
    ctot_wavefront: u.Quantity,
    *,
    pixel_scale: u.Quantity,
    angle_to_wavefront_scale: u.Quantity,
) -> None:
    """Blur ``(N, y, x)`` PSFs with ``(N, 2, 2)`` Ctot in place.

    Ctot axes are ``(x, y)``. Image columns and rows increase toward ``+x``
    and ``+y`` respectively.
    The finite-field result is not renormalized.
    """
    validate_ctot_shape(
        ctot_wavefront,
        expected_size=np.asarray(psfs).shape[0],
        label="Ctot",
    )
    psfs_array = np.asarray(psfs)
    dtype = validate_psf_dtype(psfs_array.dtype)
    if psfs_array.ndim != 3:
        raise ValueError(f"psfs must have shape (N, y, x); got {psfs_array.shape}.")
    if not np.all(np.isfinite(psfs_array)):
        raise ValueError("psfs must contain only finite values.")
    pixel_scale_value = dtype.type(
        quantity_value(pixel_scale, u.mas, label="pixel_scale").item()
    )
    scale = dtype.type(
        quantity_value(
            angle_to_wavefront_scale,
            u.nm / u.mas,
            label="angle_to_wavefront_scale",
        ).item()
    )
    if not np.isfinite(pixel_scale_value) or pixel_scale_value <= 0.0:
        raise ValueError("pixel_scale must be positive and finite.")
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("angle_to_wavefront_scale must be positive and finite.")
    input_flux = np.sum(psfs_array, axis=(-2, -1), dtype=psfs_array.dtype)
    if not np.all(np.isfinite(input_flux)) or np.any(input_flux <= 0.0):
        raise ValueError("psfs must have strictly positive finite total flux.")
    values = quantity_value(
        ctot_wavefront, u.nm**2, label="ctot_wavefront", dtype=dtype
    )
    active = np.flatnonzero(psd_valid_mask(ctot_wavefront))
    if active.size == 0:
        return
    covariance_pix2 = values / scale**2 / pixel_scale_value**2
    active_psfs = psfs_array[active].copy()
    work_shape = tuple(2 * size for size in psfs_array.shape[-2:])
    otfs = np.stack(
        [
            image_plane_gaussian_otf(covariance_pix2[index], work_shape, dtype=dtype)
            for index in active
        ]
    )
    apply_finite_fov_otfs(active_psfs, otfs)
    psfs_array[active] = active_psfs


def blur_psfs_with_ctot(
    psfs: np.ndarray,
    ctot_wavefront: u.Quantity,
    *,
    pixel_scale: u.Quantity,
    angle_to_wavefront_scale: u.Quantity,
    dtype=np.float32,
) -> np.ndarray:
    """Blur external PSFs with an already-conditioned wavefront Ctot cube.

    Return a new array without renormalizing finite-field spill. This focused
    operation does not run MASTSEL or apply a non-PSD conditioning policy.
    """
    dtype = validate_psf_dtype(dtype)
    source = np.asarray(psfs, dtype=dtype)
    if source.ndim != 3:
        raise ValueError(f"psfs must have shape (N, y, x); got {source.shape}.")
    validate_ctot_shape(ctot_wavefront, expected_size=source.shape[0], label="Ctot")
    values = quantity_value(
        ctot_wavefront, u.nm**2, label="ctot_wavefront", dtype=dtype
    )
    if not np.allclose(values, np.swapaxes(values, -1, -2)):
        raise ValueError("Ctot matrices must be symmetric.")
    if values.size and np.min(np.linalg.eigvalsh(values)) < -CTOT_PSD_TOLERANCE_NM2:
        raise ValueError(
            "Ctot must already be positive semidefinite within the fixed "
            f"{CTOT_PSD_TOLERANCE_NM2:.6g} nm2 tolerance."
        )
    result = source.copy()
    apply_ctot_blur(
        result,
        ctot_wavefront,
        pixel_scale=pixel_scale,
        angle_to_wavefront_scale=angle_to_wavefront_scale,
    )
    return result


def image_plane_gaussian_otf(
    covariance_pix2: np.ndarray,
    shape: tuple[int, int],
    *,
    mean_shift_pix: np.ndarray | None = None,
    dtype=np.float32,
) -> np.ndarray:
    """Return a finite-field Gaussian OTF for an x/y covariance in pixel squared.

    ``shape`` is the padded image shape in y/x order. ``mean_shift_pix`` is an
    optional x/y displacement in pixels. The returned array uses NumPy's
    unshifted FFT frequency convention and can be passed to
    :func:`apply_finite_fov_otfs`.
    """
    dtype = validate_psf_dtype(dtype)
    covariance_pix2 = np.asarray(covariance_pix2, dtype=dtype)
    if covariance_pix2.shape != (2, 2) or not np.all(np.isfinite(covariance_pix2)):
        raise ValueError("covariance_pix2 must be a finite 2 x 2 matrix.")
    if len(shape) != 2 or any(int(size) != size or size < 1 for size in shape):
        raise ValueError(f"shape must contain two positive integers; got {shape}.")
    mean = (
        np.zeros(2, dtype=dtype)
        if mean_shift_pix is None
        else np.asarray(mean_shift_pix, dtype=dtype)
    )
    if mean.shape != (2,) or not np.all(np.isfinite(mean)):
        raise ValueError("mean_shift_pix must contain finite x/y displacements.")
    ny, nx = shape
    fy = np.fft.fftfreq(ny).astype(dtype)
    fx = np.fft.fftfreq(nx).astype(dtype)
    fx_grid, fy_grid = np.meshgrid(fx, fy)
    covariance = 0.5 * (covariance_pix2 + covariance_pix2.T)
    exponent = (
        -2.0
        * np.pi**2
        * (
            covariance[0, 0] * fx_grid**2
            + covariance[1, 1] * fy_grid**2
            + 2.0 * covariance[0, 1] * fx_grid * fy_grid
        )
    )
    phase = -2j * np.pi * (mean[0] * fx_grid + mean[1] * fy_grid)
    return np.exp(exponent + phase)


def apply_finite_fov_otfs(psfs: np.ndarray, otfs: np.ndarray) -> None:
    """Blur a PSF cube in place with matching precomputed finite-field OTFs.

    ``psfs`` has shape ``(N, y, x)`` and ``otfs`` has shape
    ``(N, 2*y, 2*x)`` in NumPy's unshifted FFT convention. Each PSF is padded,
    blurred, cropped, and clipped for numerical negatives. Cropped flux is
    not renormalized. Invalid shapes, non-finite values, or non-positive
    retained flux raise ``ValueError``.
    """
    psfs_array = np.asarray(psfs)
    dtype = validate_psf_dtype(psfs_array.dtype)
    complex_dtype = np.complex64 if dtype.itemsize == 4 else np.complex128
    otfs_array = np.asarray(otfs, dtype=complex_dtype)
    if psfs_array.ndim != 3:
        raise ValueError(f"psfs must have shape (N, y, x); got {psfs_array.shape}.")
    expected = (psfs_array.shape[0], 2 * psfs_array.shape[1], 2 * psfs_array.shape[2])
    if otfs_array.shape != expected:
        raise ValueError(f"otfs must have shape {expected}; got {otfs_array.shape}.")
    if not np.all(np.isfinite(psfs_array)) or not np.all(np.isfinite(otfs_array)):
        raise ValueError("psfs and otfs must contain only finite values.")
    if np.any(np.sum(psfs_array, axis=(-2, -1), dtype=psfs_array.dtype) <= 0.0):
        raise ValueError("psfs must have strictly positive finite total flux.")
    work_shape = tuple(otfs_array.shape[-2:])
    original_shape = tuple(psfs_array.shape[-2:])
    for index, otf in enumerate(otfs_array):
        padded = _centered_pad(psfs_array[index], work_shape)
        raw = np.real(ifft2(fft2(padded) * otf))
        raw = _center_crop(raw, original_shape)
        raw = np.clip(raw.astype(psfs_array.dtype, copy=False), 0.0, None)
        if float(np.sum(raw, dtype=psfs_array.dtype)) <= 0.0:
            raise ValueError(
                f"Finite-FOV blur produced non-positive flux for PSF {index}."
            )
        psfs_array[index] = raw


def _centered_pad(psf: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    source_y, source_x = psf.shape
    target_y, target_x = shape
    if target_y < source_y or target_x < source_x:
        raise ValueError(
            f"Target padded shape {shape} cannot contain PSF shape {psf.shape}."
        )
    before_y = (target_y - source_y) // 2
    before_x = (target_x - source_x) // 2
    return np.pad(
        psf,
        (
            (before_y, target_y - source_y - before_y),
            (before_x, target_x - source_x - before_x),
        ),
        mode="constant",
    )


def _center_crop(psf: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    target_y, target_x = shape
    source_y, source_x = psf.shape
    if target_y > source_y or target_x > source_x:
        raise ValueError(
            f"Target crop shape {shape} cannot exceed PSF shape {psf.shape}."
        )
    start_y = (source_y - target_y) // 2
    start_x = (source_x - target_x) // 2
    return psf[start_y : start_y + target_y, start_x : start_x + target_x]


def psd_valid_mask(ctot_wavefront: u.Quantity) -> np.ndarray:
    """Return the mask of nonzero Ctot matrices that satisfy the PSD tolerance."""
    validate_ctot_shape(ctot_wavefront, label="Ctot")
    return _psd_valid_mask(ctot_wavefront)


def validate_ctot_shape(
    ctot: u.Quantity,
    *,
    label: str,
    expected_size: int | None = None,
) -> None:
    """Validate a finite Ctot covariance cube."""
    if not isinstance(ctot, u.Quantity):
        raise TypeError(f"{label} must be an Astropy Quantity.")
    if ctot.ndim != 3 or ctot.shape[-2:] != (2, 2):
        raise ValueError(f"{label} must have shape (N, 2, 2); got {ctot.shape}.")
    if expected_size is not None and ctot.shape[0] != int(expected_size):
        raise ValueError(
            f"{label} field length {ctot.shape[0]} does not match expected {expected_size}."
        )
    if not np.all(np.isfinite(ctot.value)):
        raise ValueError(f"{label} contains non-finite values.")


__all__ = [
    "apply_finite_fov_otfs",
    "blur_psfs_with_ctot",
    "image_plane_gaussian_otf",
    "simulate",
]
