"""Generate synthetic HDF5 telemetry for the E2E-Calibrated example.

Atmospheric tilt and residual coma are correlated across field directions
and frames. The coma probe uses independent wavefronts with the same
primary-coma RMS as the covariance histories.

The synthetic WFS applies flux-dependent responses to coma and imposed
spot displacements. Both measurements use the same centroid noise, with
heavier tails at low photon counts.

The reconstructor prior shares the WFS noise scale and uses spot FWHM
and the product of photon count and encircled energy.
"""

from __future__ import annotations

import argparse
from itertools import product
from pathlib import Path

import h5py
import numpy as np
from ao_core.io.ini import (
    read_ini,
    read_ini_array,
    read_ini_scalar,
)
from astropy import units as u

from hybrid_ao_psf import (
    CalibrationPhotometry,
    ComaWavefronts,
    ComaWfsMeasurements,
    load_e2e_calibration_configuration,
    load_psf_metric_interpolator,
)

FIELD_POSITIONS = np.array(list(product((-60, 0, 60), repeat=2))) * u.arcsec
COMA_MAGNITUDES = (12, 14, 16, 18, 20) * u.mag
SENSING_MAGNITUDES = (14, 16, 18) * u.mag
PRIOR_SIGNAL_LIMITS = (0.01, 1e4) * u.photon
PRIOR_FWHM_LIMITS = (30, 450) * u.mas
PRIOR_SIGNAL_POINTS = 33
PRIOR_FWHM_POINTS = 33

PUPIL_RESOLUTION = 32
COMA_SAMPLES = 256
COMA_NOISE_DRAWS = 2048
SENSING_ZERO_DRAWS = 2048
SENSING_GAIN_DRAWS = 256
SENSING_DISPLACEMENT = 10 * u.mas

RANDOM_SEED = 42
REALIZATIONS = 4
FRAMES_PER_REALIZATION = 600
ATMOSPHERIC_TILT_RMS_MAS = 50
TILT_ANGULAR_SCALE_ARCSEC = 200
TILT_TIME_SCALE_S = 0.45
RESIDUAL_COMA_RMS_NM = 70
COMA_ANGULAR_SCALE_ARCSEC = 40
COMA_TIME_SCALE_S = 0.10
TILT_COMA_CORRELATION = -0.15

HALF_RESPONSE_PHOTONS_PER_FRAME = 2.5
ZENITH_COMA_RESPONSE_MAS_PER_NM = 0.1
COMA_RESPONSE_PER_AIRMASS_MAS_PER_NM = 0.07
ZENITH_ON_AXIS_GAIN = 0.9
GAIN_CHANGE_AT_FIELD_EDGE = 0.1
GAIN_PER_AIRMASS = 0.3
NOISE_SCALE_MAS = 230
WIDE_NOISE_MAX_PROBABILITY = 0.1
WIDE_NOISE_FLUX_SCALE = 10
WIDE_NOISE_WIDTH_FACTOR = 2

PRIOR_WIDTH_SCALE_MAS = 240
PRIOR_PHOTON_OFFSET = 2
PRIOR_LOW_FLUX_OFFSET = 0.6


def _write_telemetry(path, dtype, **datasets):
    with h5py.File(path, "w") as telemetry:
        for name, values in datasets.items():
            telemetry[name] = np.asarray(values, dtype=dtype)


def _gaussian_correlation_factor(separations, scale):
    """Factor the Gaussian correlation matrix, clipping negative roundoff."""
    correlation = np.exp(-((separations / scale) ** 2))
    eigenvalues, eigenvectors = np.linalg.eigh(correlation)
    return eigenvectors * np.sqrt(np.maximum(eigenvalues, 0))


def _gaussian_field(rng, frame_lags, field_separations, *, time_scale, angular_scale):
    """Draw a unit-RMS field with separate temporal and angular correlations."""
    time_factor = _gaussian_correlation_factor(frame_lags, time_scale)
    field_factor = _gaussian_correlation_factor(field_separations, angular_scale)
    # Draw one independent history for each realization and tip/tilt component.
    noise = rng.normal(
        size=(REALIZATIONS, 2, FRAMES_PER_REALIZATION, len(FIELD_POSITIONS))
    )
    field = time_factor @ noise @ field_factor.T
    return field.transpose(0, 2, 3, 1)


def _tilt_and_coma(rng, frame_rate_hz):
    """Generate paired tilt and residual coma in mas and nm OPD."""
    frame_times = np.arange(FRAMES_PER_REALIZATION) / frame_rate_hz
    frame_lags = np.subtract.outer(frame_times, frame_times)
    positions = FIELD_POSITIONS.to_value(u.arcsec)
    field_separations = np.linalg.norm(positions[:, None] - positions[None, :], axis=2)

    tilt_field = _gaussian_field(
        rng,
        frame_lags,
        field_separations,
        time_scale=TILT_TIME_SCALE_S,
        angular_scale=TILT_ANGULAR_SCALE_ARCSEC,
    )
    coma_field = _gaussian_field(
        rng,
        frame_lags,
        field_separations,
        time_scale=COMA_TIME_SCALE_S,
        angular_scale=COMA_ANGULAR_SCALE_ARCSEC,
    )
    tilt = ATMOSPHERIC_TILT_RMS_MAS * tilt_field
    coma = RESIDUAL_COMA_RMS_NM * (
        TILT_COMA_CORRELATION * tilt_field
        + np.sqrt(1 - TILT_COMA_CORRELATION**2) * coma_field
    )
    return tilt, coma


def _coma_wavefronts(sample_count, pupil_grid, rng, dtype):
    """Draw primary coma and construct the corresponding annular-pupil OPD."""
    x, y, pupil = pupil_grid

    # Remove the projection onto tilt and normalize each mode to unit pupil RMS.
    tilt = np.stack((x, y))[:, pupil]
    radial_term = x[pupil] ** 2 + y[pupil] ** 2
    projection = np.sum(tilt**2 * radial_term, axis=1) / np.sum(tilt**2, axis=1)
    modes = tilt * (radial_term - projection[:, None])
    modes /= np.sqrt(np.mean(modes**2, axis=1, keepdims=True))

    coma = rng.normal(scale=RESIDUAL_COMA_RMS_NM, size=(sample_count, 2))
    wavefront = np.zeros((sample_count, *pupil.shape))
    wavefront[:, pupil] = coma @ modes
    return ComaWavefronts(
        wavefront=wavefront * u.nm, coma=coma * u.nm, pupil=pupil, dtype=dtype
    )


def _relative_response(photons_per_frame):
    """Return the fraction of the bright-star response at this photon count."""
    return photons_per_frame / (photons_per_frame + HALF_RESPONSE_PHOTONS_PER_FRAME)


def _centroid_noise(rng, photons_per_frame, draw_shape):
    """Draw two-axis noise with a shared width and RMS 230/sqrt(P) per axis."""
    wide_probability = WIDE_NOISE_MAX_PROBABILITY / (
        1 + (photons_per_frame / WIDE_NOISE_FLUX_SCALE) ** 2
    )
    is_wide = rng.random((*draw_shape, 1)) < wide_probability
    width = np.where(is_wide, WIDE_NOISE_WIDTH_FACTOR, 1)
    mixture_rms = np.sqrt(1 + wide_probability * (WIDE_NOISE_WIDTH_FACTOR**2 - 1))
    noise_rms_mas = NOISE_SCALE_MAS / np.sqrt(photons_per_frame)
    return noise_rms_mas * (width / mixture_rms) * rng.normal(size=(*draw_shape, 2))


def _measure_coma(zenith_angle, wavefronts, photon_counts, settings, dtype):
    """Apply the coma response and repeat the noisy centroid measurement."""
    airmass = 1 / np.cos(zenith_angle.to_value(u.rad))
    response = (
        ZENITH_COMA_RESPONSE_MAS_PER_NM
        + COMA_RESPONSE_PER_AIRMASS_MAS_PER_NM * (airmass - 1)
    ) * _relative_response(photon_counts)
    noiseless = wavefronts.coma.value[:, None, :] * response[None, :, None]

    rng = np.random.default_rng([settings.noise_seed, int(zenith_angle.value)])
    samples = len(wavefronts.coma)
    measured = np.empty(
        (samples, len(photon_counts), settings.noise_draws, 2), dtype=dtype
    )
    for magnitude, photons_per_frame in enumerate(photon_counts):
        noise = _centroid_noise(rng, photons_per_frame, (samples, settings.noise_draws))
        measured[:, magnitude] = noiseless[:, magnitude, None, :] + noise
    return ComaWfsMeasurements(
        measured=measured * u.mas,
        noiseless=noiseless * u.mas,
        dtype=dtype,
    )


def _measure_fixed_spot(
    field_fraction, airmass, photons_per_frame, shifts, settings, seed
):
    """Simulate an unshifted spot and shifted reads with paired noise."""
    gain = (
        ZENITH_ON_AXIS_GAIN
        + GAIN_CHANGE_AT_FIELD_EDGE * field_fraction
        + GAIN_PER_AIRMASS * (airmass - 1)
    ) * _relative_response(photons_per_frame)

    rng = np.random.default_rng(seed)
    unshifted = _centroid_noise(rng, photons_per_frame, (settings.zero_draws,))
    paired_noise = _centroid_noise(rng, photons_per_frame, (settings.gain_draws,))
    shifted = gain * shifts[:, None, :] + paired_noise[None, :, :]
    return unshifted, shifted


def _write_covariance(path, rng, frame_rate_hz, dtype):
    tilt, coma = _tilt_and_coma(rng, frame_rate_hz)
    _write_telemetry(path, dtype, positions=FIELD_POSITIONS.value, tilt=tilt, coma=coma)


def _write_coma(path, zenith_angle, rng, config, photometry, pupil_grid):
    photon_counts = photometry.magnitude_to_photons_per_frame(COMA_MAGNITUDES).value
    wavefronts = _coma_wavefronts(config.coma.samples, pupil_grid, rng, config.dtype)
    coma_wfs = _measure_coma(
        zenith_angle, wavefronts, photon_counts, config.coma, config.dtype
    )
    _write_telemetry(
        path,
        config.dtype,
        magnitudes=COMA_MAGNITUDES.value,
        wavefront=wavefronts.wavefront.value,
        coma=wavefronts.coma.value,
        tilt_noiseless=coma_wfs.noiseless.value,
        tilt=coma_wfs.measured.value,
    )


def _write_sensing(
    path, zenith_angle, plane, config, photometry, field_radius, ngs_metrics
):
    positions = FIELD_POSITIONS
    magnitudes = SENSING_MAGNITUDES
    metrics = ngs_metrics.evaluate(
        zenith_angle=zenith_angle, x=positions[:, 0], y=positions[:, 1]
    )
    photon_counts = photometry.magnitude_to_photons_per_frame(magnitudes).value
    airmass = 1 / np.cos(zenith_angle.to_value(u.rad))
    shifts = config.sensing.displacement.to_value(u.mas) * np.array(
        [[1, 0], [-1, 0], [0, 1], [0, -1]]
    )
    grid_shape = (len(positions), len(magnitudes))
    tilt_unshifted = np.empty(
        (*grid_shape, config.sensing.zero_draws, 2), dtype=config.dtype
    )
    tilt_shifted = np.empty(
        (*grid_shape, len(shifts), config.sensing.gain_draws, 2), dtype=config.dtype
    )
    first_seed = config.sensing.seed + plane * np.prod(grid_shape)
    for case, (direction, magnitude) in enumerate(np.ndindex(grid_shape)):
        tilt_unshifted[direction, magnitude], tilt_shifted[direction, magnitude] = (
            _measure_fixed_spot(
                np.linalg.norm(positions[direction] / field_radius).value,
                airmass,
                photon_counts[magnitude],
                shifts,
                config.sensing,
                first_seed + case,
            )
        )
    _write_telemetry(
        path,
        config.dtype,
        positions=positions.value,
        magnitudes=magnitudes.value,
        ee=metrics.ee.value,
        fwhm=metrics.fwhm.to_value(u.mas),
        shifts=shifts,
        tilt_unshifted=tilt_unshifted,
        tilt_shifted=tilt_shifted,
    )


def _reconstructor_covariance(signal, fwhm_mas):
    """Return isotropic centroid noise at the supplied signal and spot width."""
    photon_term = np.exp(-((PRIOR_WIDTH_SCALE_MAS / fwhm_mas) ** 2)) / (
        signal + PRIOR_PHOTON_OFFSET
    )
    low_flux_term = 1 / (signal + PRIOR_LOW_FLUX_OFFSET) ** 2
    variance = NOISE_SCALE_MAS**2 * (photon_term + low_flux_term)
    return variance[..., None, None] * np.eye(2)


def _write_reconstructor(output_dir):
    """Write a measured grid and separate interpolation and equal-signal checks."""
    signal_axis = np.geomspace(
        *PRIOR_SIGNAL_LIMITS.to_value(u.photon), PRIOR_SIGNAL_POINTS
    )
    fwhm_axis = np.geomspace(*PRIOR_FWHM_LIMITS.to_value(u.mas), PRIOR_FWHM_POINTS)
    grid_signal, grid_fwhm = np.meshgrid(signal_axis, fwhm_axis, indexing="ij")

    # Check interpolation at the centres of the log-spaced cells.
    check_signal, check_fwhm = np.meshgrid(
        np.sqrt(signal_axis[:-1] * signal_axis[1:]),
        np.sqrt(fwhm_axis[:-1] * fwhm_axis[1:]),
        indexing="ij",
    )
    # Change flux and EE together to keep their product fixed in each triple.
    equal_signal, equal_fwhm, equal_ee = np.meshgrid(
        np.geomspace(signal_axis[0], signal_axis[-1], 6),
        np.geomspace(fwhm_axis[0], fwhm_axis[-1], 4),
        [0.2, 0.5, 1.0],
        indexing="ij",
    )
    flux = np.r_[check_signal.ravel(), (equal_signal / equal_ee).ravel()]
    ee = np.r_[np.ones(check_signal.size), equal_ee.ravel()]
    fwhm = np.r_[check_fwhm.ravel(), equal_fwhm.ravel()]

    with h5py.File(output_dir / "reconstructor.h5", "w") as telemetry:
        grid = telemetry.create_group("grid")
        grid["signal"] = signal_axis
        grid["fwhm"] = fwhm_axis
        grid["covariance"] = _reconstructor_covariance(grid_signal, grid_fwhm)

        checks = telemetry.create_group("checks")
        checks["flux"] = flux
        checks["ee"] = ee
        checks["fwhm"] = fwhm
        checks["covariance"] = _reconstructor_covariance(flux * ee, fwhm)


def main(ini: Path, ngs_metrics_path: Path, output_dir: Path) -> None:
    """Write one reconstructor file and three telemetry files per zenith angle.

    Use the INI for system and sampling settings and the NGS metric
    interpolator for the sensing spots' EE and FWHM. Replace files with the
    same names on subsequent runs.
    """
    config = load_e2e_calibration_configuration(ini).with_overrides(
        dtype=np.float64,
        coma={"samples": COMA_SAMPLES, "noise_draws": COMA_NOISE_DRAWS},
        sensing={
            "zero_draws": SENSING_ZERO_DRAWS,
            "gain_draws": SENSING_GAIN_DRAWS,
            "displacement": SENSING_DISPLACEMENT,
        },
    )
    _, sections = read_ini(config.source)
    system = sections["system"]
    zenith_angles = (
        read_ini_array(system, "ZenithAngle", label="system.ZenithAngle") * u.deg
    )
    field_radius = system.getfloat("FieldRadius") * u.arcsec
    frame_rate_hz = read_ini_scalar(
        system, "SensorFrameRate_LO", label="system.SensorFrameRate_LO"
    )
    if frame_rate_hz <= 0:
        raise ValueError("[system] SensorFrameRate_LO must be positive.")
    photometry = CalibrationPhotometry(
        magnitude_band=system["NGSMagStandard"],
        photons_per_frame_zeropoint=(system.getfloat("NGSMagZeropoint") * u.photon),
        dtype=config.dtype,
    )
    obscuration_ratio = read_ini_scalar(
        system, "ObscurationRatio", label="system.ObscurationRatio"
    )
    if not 0 <= obscuration_ratio < 1:
        raise ValueError("[system] ObscurationRatio must be in [0, 1).")
    coords = np.linspace(-1, 1, PUPIL_RESOLUTION)
    x, y = np.meshgrid(coords, coords)
    radius_squared = x**2 + y**2
    pupil = (radius_squared <= 1) & (radius_squared >= obscuration_ratio**2)

    ngs_metrics = load_psf_metric_interpolator(ngs_metrics_path)
    rng = np.random.default_rng(RANDOM_SEED)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_reconstructor(output_dir)
    for plane, zenith_angle in enumerate(zenith_angles):
        angle = np.format_float_positional(zenith_angle.to_value(u.deg), trim="-")
        _write_covariance(
            output_dir / f"covariance-za{angle}.h5", rng, frame_rate_hz, config.dtype
        )
        _write_coma(
            output_dir / f"coma-za{angle}.h5",
            zenith_angle,
            rng,
            config,
            photometry,
            (x, y, pupil),
        )
        _write_sensing(
            output_dir / f"sensing-za{angle}.h5",
            zenith_angle,
            plane,
            config,
            photometry,
            field_radius,
            ngs_metrics,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("ini", type=Path, help="Example calibration INI.")
    parser.add_argument(
        "--ngs-metrics",
        type=Path,
        required=True,
        help="NGS PSF metric interpolator file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for the generated HDF5 files.",
    )
    args = parser.parse_args()
    main(args.ini, args.ngs_metrics, args.output_dir)
