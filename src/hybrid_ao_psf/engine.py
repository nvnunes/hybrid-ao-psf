"""Stateless execution of one science field.

Evaluate the shared science PSF provider, use the selected low-order
model and apply its displacement distribution with finite-field blur.
The public entry points follow the execution and numerical helpers.
"""

from __future__ import annotations

from configparser import ConfigParser
from dataclasses import replace
from io import StringIO

import numpy as np
from ao_core.photometry import (
    magnitudes_to_effective_photons_per_second,
)
from astropy import units as u
from numpy.typing import DTypeLike
from scipy.fft import fft2, ifft2

from ._coordinates import _cartesian_to_polar
from ._e2e_model import E2ECalibratedModel
from ._mastsel import (
    MastselModel,
    _capture_mastsel,
    _collect_mastsel_statistics,
    _compute_mastsel_ctot,
    _mastsel_angle_to_wavefront_scale,
    _mastsel_frame_rate,
    _mastsel_loop_settings,
    _mastsel_photometry,
    _MastselPrediction,
    _open_mastsel,
)
from ._units import quantity_value
from ._validation import validate_psf_dtype
from .contracts import (
    CTOT_PSD_TOLERANCE_NM2,
    CovarianceDiagnostics,
    DiagnosticsLevel,
    HybridCtotContext,
    HybridCtotCorrection,
    HybridDiagnostics,
    HybridRequest,
    HybridResult,
    LowOrderDistribution,
    MastselDiagnostics,
    NonPsdPolicy,
    PsfMetricPrediction,
    PsfPrediction,
    PsfProvider,
    _displacement_psd_mask,
    _jitter_from_ctot,
    _positive_scalar,
    _psd_valid_mask,
)
from .interpolation import PsfInterpolator, PsfMetricInterpolator
from .psf_files import PsfFileProvider

# Shared simulation setup


def _validate_simulation_inputs(
    request: HybridRequest,
    science_psfs: PsfProvider,
    lo_model: MastselModel | E2ECalibratedModel,
) -> None:
    """Check the observation against the selected model before evaluation."""
    if not isinstance(request, HybridRequest):
        raise TypeError("request must be a HybridRequest.")
    if not isinstance(science_psfs, PsfProvider):
        raise TypeError("science_psfs must implement PsfProvider.")
    if isinstance(lo_model, MastselModel):
        if (
            request.ngs_magnitude is not None
            and lo_model.ngs_magnitude_zeropoint is None
        ):
            raise ValueError("MASTSEL magnitudes require ngs_magnitude_zeropoint.")
    elif isinstance(lo_model, E2ECalibratedModel):
        if request.ngs_magnitude is None:
            raise ValueError("E2E-calibrated prediction requires ngs_magnitude.")
    else:
        raise TypeError("lo_model must be a MastselModel or E2ECalibratedModel.")


def _provider_at_precision[Provider](provider: Provider, dtype: np.dtype) -> Provider:
    """Convert built-in PSF sources internally at the request's precision."""
    if (
        isinstance(provider, (PsfInterpolator, PsfMetricInterpolator, PsfFileProvider))
        and provider.dtype != dtype
    ):
        return provider._with_dtype(dtype)
    return provider


def _evaluate_science_psfs(
    request: HybridRequest, science_psfs: PsfProvider
) -> PsfPrediction:
    science = science_psfs.evaluate(
        zenith_angle=request.zenith_angle,
        wavelength=request.wavelength,
        x=request.science_positions[:, 0],
        y=request.science_positions[:, 1],
    )
    if science.psfs.shape[0] != len(request.science_positions):
        raise ValueError("Science provider output length does not match the request.")
    return science


# MASTSEL simulation


def _simulate_mastsel(
    request: HybridRequest,
    science: PsfPrediction,
    model: MastselModel,
) -> HybridResult:
    """Evaluate the MASTSEL covariance and apply its conditioned Gaussian blur."""
    prediction = _predict_mastsel(request, model)
    psfs = np.array(science.psfs, dtype=request.dtype, copy=True)
    apply_ctot_blur(
        psfs,
        prediction.ctot_wavefront,
        pixel_scale=science.pixel_scale,
        angle_to_wavefront_scale=prediction.angle_to_wavefront_scale,
    )
    diagnostics = None
    if request.diagnostics_level is not DiagnosticsLevel.NONE:
        debug = request.diagnostics_level is DiagnosticsLevel.DEBUG
        diagnostics = HybridDiagnostics(
            covariance=CovarianceDiagnostics(
                ctot=prediction.ctot,
                psd_valid_mask=_displacement_psd_mask(
                    prediction.ctot, prediction.ctot_wavefront
                ),
                psd_clipped=prediction.psd_clipped,
            ),
            mastsel=MastselDiagnostics(
                ngs_metrics=prediction.ngs_metrics,
                ngs_flux=prediction.ngs_flux,
                frame_rate=prediction.frame_rate,
                angle_to_wavefront_scale=prediction.angle_to_wavefront_scale,
                effective_ini=prediction.effective_ini if debug else None,
            ),
            psfs_ho=science.psfs if debug else None,
        )
    return HybridResult(
        positions=request.science_positions,
        psfs=psfs,
        jitter=_jitter_from_ctot(prediction.ctot),
        metadata=science.metadata,
        diagnostics=diagnostics,
    )


def _predict_mastsel(
    request: HybridRequest, model: MastselModel
) -> _MastselPrediction:
    """Evaluate and condition MASTSEL covariance at the requested precision."""
    ngs_metrics = _provider_at_precision(model.ngs_metrics, request.dtype)
    metrics = ngs_metrics.evaluate(
        zenith_angle=request.zenith_angle,
        x=request.ngs_positions[:, 0],
        y=request.ngs_positions[:, 1],
    )
    if metrics.ee.size != len(request.ngs_positions):
        raise ValueError("NGS provider output length does not match the request.")

    metrics = replace(metrics, dtype=request.dtype)
    parser, lo_frame_rate, ngs_flux, zeropoint = _resolve_effective_mastsel(
        request, model
    )
    stream = StringIO()
    parser.write(stream)
    effective_ini = stream.getvalue()
    ctot_raw, angle_to_wavefront_scale = _predict_mastsel_ctot(
        request,
        metrics,
        parser,
        effective_ini,
        lo_frame_rate,
        ngs_flux,
        zeropoint,
        model.ctot_correction,
    )
    ctot_wavefront, psd_clipped = condition_ctot(
        ctot_raw,
        policy=model.non_psd_policy,
        dtype=request.dtype,
    )
    angle_to_wavefront_scale = u.Quantity(angle_to_wavefront_scale, dtype=request.dtype)
    ctot_angle = ctot_wavefront / angle_to_wavefront_scale**2
    return _MastselPrediction(
        ctot=ctot_angle,
        ctot_wavefront=ctot_wavefront,
        angle_to_wavefront_scale=angle_to_wavefront_scale,
        psd_clipped=psd_clipped,
        ngs_metrics=metrics,
        ngs_flux=ngs_flux,
        frame_rate=lo_frame_rate,
        effective_ini=effective_ini,
    )


def _predict_mastsel_ctot(
    request: HybridRequest,
    metrics: PsfMetricPrediction,
    parser: ConfigParser,
    effective_ini: str,
    lo_frame_rate: u.Quantity,
    ngs_flux: u.Quantity,
    ngs_magnitude_zeropoint: u.Quantity | None,
    ctot_correction: HybridCtotCorrection | None,
) -> tuple[u.Quantity, u.Quantity]:
    """Use the live runtime for covariance prediction and the optional callback."""
    with _open_mastsel(parser, effective_ini) as model:
        angle_to_wavefront_scale = _mastsel_angle_to_wavefront_scale(model)
        if ctot_correction is None:
            ctot_raw = _compute_mastsel_ctot(
                model, request, metrics, lo_frame_rate, ngs_flux
            )
        else:
            with _capture_mastsel(model) as captured:
                ctot_raw = _compute_mastsel_ctot(
                    model, request, metrics, lo_frame_rate, ngs_flux
                )
            statistics = _collect_mastsel_statistics(
                model, captured, request, lo_frame_rate, angle_to_wavefront_scale
            )
        validate_ctot_shape(
            ctot_raw, expected_size=len(request.science_positions), label="MASTSEL Ctot"
        )
        ctot_raw = u.Quantity(ctot_raw, dtype=request.dtype)
        if ctot_correction is not None:
            loop_gain, loop_delay = _mastsel_loop_settings(parser)
            context = HybridCtotContext(
                request=request,
                ngs_metrics=metrics,
                mastsel_statistics=statistics,
                angle_to_wavefront_scale=angle_to_wavefront_scale,
                lo_frame_rate=lo_frame_rate,
                loop_gain=loop_gain,
                loop_delay=loop_delay,
                ngs_magnitude_zeropoint=ngs_magnitude_zeropoint,
            )
            ctot_raw.flags.writeable = False
            corrected = ctot_correction.apply(ctot_raw, context)
            if corrected is not None:
                validate_ctot_shape(
                    corrected,
                    expected_size=len(request.science_positions),
                    label="Corrected Ctot",
                )
                ctot_raw = corrected.to(u.nm**2)
    return ctot_raw, angle_to_wavefront_scale


# E2E-Calibrated simulation


def _simulate_calibrated(
    request: HybridRequest,
    science: PsfPrediction,
    model: E2ECalibratedModel,
) -> HybridResult:
    """Apply the validated calibrated distribution without extra conditioning."""
    model = model.prepare(dtype=request.dtype)
    prediction = model.predict(
        ngs_positions=request.ngs_positions,
        ngs_magnitude=request.ngs_magnitude,
        science_positions=request.science_positions,
        zenith_angle=request.zenith_angle,
        diagnostics_level=request.diagnostics_level,
    )
    distribution = prediction.displacement_distribution
    psfs = np.array(science.psfs, dtype=request.dtype, copy=True)
    _apply_displacement_mixture(psfs, distribution, science.pixel_scale, request.dtype)
    total_covariance = distribution.covariance
    diagnostics = None
    if request.diagnostics_level is not DiagnosticsLevel.NONE:
        debug = request.diagnostics_level is DiagnosticsLevel.DEBUG
        diagnostics = HybridDiagnostics(
            covariance=CovarianceDiagnostics(
                ctot=total_covariance,
                psd_valid_mask=_displacement_psd_mask(total_covariance),
            ),
            e2e_calibrated=prediction if debug else None,
            psfs_ho=science.psfs if debug else None,
        )
    return HybridResult(
        positions=request.science_positions,
        psfs=psfs,
        jitter=_jitter_from_ctot(total_covariance),
        metadata=science.metadata,
        diagnostics=diagnostics,
    )


def _apply_displacement_mixture(
    psfs: np.ndarray,
    distribution: LowOrderDistribution,
    pixel_scale: u.Quantity,
    dtype: np.dtype,
) -> None:
    """Apply the fitted angular distribution on the padded PSF grid in place."""
    work_shape = tuple(2 * size for size in psfs.shape[-2:])
    pixel = _positive_scalar(pixel_scale, u.mas, "pixel_scale", dtype=dtype).value
    otfs = np.stack(
        [
            sum(
                weight
                * image_plane_gaussian_otf(
                    covariance / pixel**2, work_shape, dtype=dtype
                )
                for weight, covariance in zip(
                    distribution.weights, matrices, strict=True
                )
            )
            for matrices in distribution.covariances.value
        ]
    )
    apply_finite_fov_otfs(psfs, otfs)


# Covariance conditioning


def condition_ctot(
    ctot_wavefront: u.Quantity,
    *,
    policy: NonPsdPolicy | str = NonPsdPolicy.ERROR,
    dtype: DTypeLike = np.float32,
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


# MASTSEL configuration and photometry


def _resolve_effective_mastsel(
    request: HybridRequest,
    model: MastselModel,
) -> tuple[ConfigParser, u.Quantity, u.Quantity, u.Quantity | None]:
    """Resolve MASTSEL options, frame rate, effective flux and its zeropoint."""
    parser = model._copy_ini()
    for section in ("telescope", "sources_LO"):
        if not parser.has_section(section):
            raise ValueError(f"MASTSEL configuration requires [{section}].")
    lo_frame_rate = _mastsel_frame_rate(parser)
    if not parser.has_section("sources_science"):
        parser.add_section("sources_science")
    parser["telescope"]["ZenithAngle"] = _format_scalar(
        request.zenith_angle.to_value(u.deg)
    )
    science_r, science_theta = _cartesian_to_polar(request.science_positions)
    ngs_r, ngs_theta = _cartesian_to_polar(request.ngs_positions)
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
            np.full(len(request.ngs_positions), lenslets[0], dtype=float)
        )
    zeropoint = None
    if request.ngs_flux is not None:
        ngs_flux = request.ngs_flux
    else:
        zeropoint = _positive_scalar(
            model.ngs_magnitude_zeropoint,
            u.photon / (u.m**2 * u.s),
            "ngs_magnitude_zeropoint",
            dtype=request.dtype,
        )
        photometry = _mastsel_photometry(parser, lo_frame_rate, zeropoint)
        ngs_flux = magnitudes_to_effective_photons_per_second(
            request.ngs_magnitude, photometry
        )
        if np.any(ngs_flux.to_value(u.photon / u.s) <= 0.0):
            raise ValueError("ngs_magnitude resolves to non-positive ngs_flux.")
    return parser, lo_frame_rate, ngs_flux, zeropoint


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


# Finite-field convolution


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
    The finite-field result is not renormalized. If a blur fails partway
    through the cube, earlier images may already be updated.
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
    work_shape = tuple(2 * size for size in psfs_array.shape[-2:])
    for index in active:
        otf = image_plane_gaussian_otf(covariance_pix2[index], work_shape, dtype=dtype)
        apply_finite_fov_otfs(psfs_array[index : index + 1], otf[None])


def _validated_otf_covariance(values: np.ndarray, dtype: np.dtype) -> np.ndarray:
    """Accept a physically valid pixel covariance with dtype-scale round-off."""
    covariance = np.asarray(values)
    if covariance.shape != (2, 2) or covariance.dtype.kind not in "fiu":
        raise ValueError("covariance_pix2 must be a finite real 2 x 2 matrix.")
    covariance = covariance.astype(dtype, copy=False)
    if not np.all(np.isfinite(covariance)):
        raise ValueError("covariance_pix2 must be a finite real 2 x 2 matrix.")
    scale = float(np.max(np.abs(covariance)))
    tolerance = 8 * np.finfo(dtype).eps * scale
    if not np.allclose(covariance, covariance.T, rtol=0, atol=tolerance):
        raise ValueError("covariance_pix2 must be symmetric.")
    covariance = 0.5 * (covariance + covariance.T)
    eigenvalues = np.linalg.eigvalsh(covariance)
    if eigenvalues[0] < -tolerance:
        raise ValueError("covariance_pix2 must be positive semidefinite.")
    if eigenvalues[0] < 0:
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        covariance = (eigenvectors * np.maximum(eigenvalues, 0)) @ eigenvectors.T
    return covariance


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


# Public simulation and image operations


def image_plane_gaussian_otf(
    covariance_pix2: np.ndarray,
    shape: tuple[int, int],
    *,
    mean_shift_pix: np.ndarray | None = None,
    dtype: DTypeLike = np.float32,
) -> np.ndarray:
    """Sample the Gaussian optical transfer function on the padded FFT grid.

    Args:
        covariance_pix2: Finite ``(2, 2)`` covariance in squared pixels, with both
            axes in x/y order. The calculation uses its symmetric part.
        shape: Number of rows and columns in the padded image.
        mean_shift_pix: Optional finite x/y displacement in pixels. The
            default uses a centred Gaussian.
        dtype: Real calculation precision, float32 or float64.

    Returns:
        Complex OTF sampled at NumPy's unshifted FFT frequencies, suitable for
        [`apply_finite_fov_otfs()`][hybrid_ao_psf.apply_finite_fov_otfs].

    Raises:
        ValueError: If the dimensions, covariance, displacement or dtype
            are invalid.
    """
    dtype = validate_psf_dtype(dtype)
    covariance = _validated_otf_covariance(covariance_pix2, dtype)
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
    """Pad, blur and crop each image in place using its supplied OTF.

    Clip numerical negatives after cropping. The retained flux can differ
    from the input flux because light can leave the output field. If a blur
    fails partway through the cube, earlier images may already be updated.

    Args:
        psfs: Writable float32 or float64 cube with shape ``(N, y, x)``.
        otfs: Matching OTFs of shape ``(N, 2*y, 2*x)`` in NumPy's unshifted
            FFT frequency order.

    Raises:
        ValueError: If shapes, precision, values or retained flux are invalid,
            or if the input cube is read-only.
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


def blur_psfs_with_ctot(
    psfs: np.ndarray,
    ctot_wavefront: u.Quantity,
    *,
    pixel_scale: u.Quantity,
    angle_to_wavefront_scale: u.Quantity,
    dtype: DTypeLike = np.float32,
) -> np.ndarray:
    """Blur PSFs with supplied, already-conditioned wavefront covariance.

    Use a Gaussian kernel derived from the angular covariance and crop to
    the original image. The returned flux reflects any light lost in the crop.

    Args:
        psfs: Source image cube in ``(points, rows, columns)`` order.
        ctot_wavefront: Symmetric, positive semidefinite ``(points, 2, 2)``
            covariance in square nanometres, with both matrix axes in x/y order.
        pixel_scale: Angular pixel scale, compatible with milliarcseconds.
        angle_to_wavefront_scale: Scalar conversion from milliarcseconds
            to nanometres of wavefront tilt, in nanometres per milliarcsecond.
        dtype: Pixel calculation and output precision, float32 or float64.

    Returns:
        A new image cube with the same shape as the input.

    Raises:
        TypeError: If a physical input lacks an Astropy quantity.
        ValueError: If precision, shapes, units, covariance or flux are invalid.
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


def simulate(
    request: HybridRequest,
    science_psfs: PsfProvider,
    *,
    lo_model: MastselModel | E2ECalibratedModel,
) -> HybridResult:
    """Run the selected low-order model and blur one field of science PSFs.

    The science provider determines the sampling and flux convention. Both
    low-order models use the same finite-field convolution. E2E-Calibrated
    uses the full calibrated two-Gaussian mixture. The operation borrows the
    providers, prepares them for the request dtype when needed, and returns
    independent result arrays.

    Args:
        request: Science field and asterism for one observation. Supply
            effective photon rates or magnitudes for MASTSEL, and magnitudes
            in the calibrated convention for E2E-Calibrated.
        science_psfs: Image provider evaluated at the science coordinates,
            zenith angle and wavelength of the request.
        lo_model: Configured MASTSEL or E2E-Calibrated model, including its NGS
            provider and photometry settings.

    Returns:
        Field positions, final science PSFs, jitter and requested diagnostics.

    Raises:
        TypeError: If the request or required providers have incompatible types.
        ValueError: If inputs conflict with the selected model, are outside
            provider support or produce invalid scientific results.
    """
    _validate_simulation_inputs(request, science_psfs, lo_model)
    science_psfs = _provider_at_precision(science_psfs, request.dtype)
    science = _evaluate_science_psfs(request, science_psfs)
    if isinstance(lo_model, MastselModel):
        return _simulate_mastsel(request, science, lo_model)
    return _simulate_calibrated(request, science, lo_model)


__all__ = [
    "apply_finite_fov_otfs",
    "blur_psfs_with_ctot",
    "image_plane_gaussian_otf",
    "simulate",
]
