"""Generate synthetic calibration files for the E2E-Calibrated example.

The files contain illustrative coma and reconstructor noise models,
covariances from synthetic tilt and coma histories, and a sensing model
built from simulated WFS tilt samples.
"""

import argparse
from collections.abc import Iterator
from itertools import product
from pathlib import Path

import numpy as np
from astropy import units as u

from hybrid_ao_psf import (
    CalibrationPhotometry,
    ComaResponseModel,
    CovarianceCalibrationSettings,
    CovarianceFunctions,
    CovarianceTelemetry,
    FixedGainIntegrator,
    NoiseBlurModel,
    ReconstructorNoiseModel,
    SensingCalibrationMeasurements,
    SensingModel,
    SensingWfsMeasurements,
    SensingWfsSampling,
    build_covariance_calibration,
    build_sensing_response,
    measure_covariance_calibration,
    save_coma_calibration,
    save_covariance_calibration,
    save_reconstructor_calibration,
    save_sensing_calibration,
)

# Shared synthetic setup.
DTYPE = np.float32
RANDOM_SEED = 42
ZENITH_ANGLE = 20 * u.deg
FIELD_AXIS_ARCSEC = (-60, 0, 60)
MAGNITUDES = (14, 18)
LO_GAIN = 0.3
LO_DELAY_FRAMES = 2
LO_FRAME_RATE = 500 * u.Hz
MAGNITUDE_BAND = "synthetic"
PHOTON_ZEROPOINT = 1e8 * u.photon

# Reconstructor noise prior.
RECONSTRUCTOR_PHOTON_GRID = (1, 1000)
RECONSTRUCTOR_FWHM_GRID_MAS = (50, 250)
RECONSTRUCTOR_VARIANCE_MAS2 = 9

# Synthetic tilt and coma histories.
COVARIANCE_FEEDBACK_STRENGTHS = (0.2, 0.6, 1.0, 1.4)
REALIZATIONS = 4
FRAMES_PER_REALIZATION = 600
SHARED_TILT_STD_MAS = 25
LOCAL_TILT_STD_MAS = 8
SHARED_COMA_STD_NM = 4
LOCAL_COMA_STD_NM = 3

# Coma response.
ZENITH_COMA_RESPONSE = 0.08 * u.mas / u.nm
COMA_RESPONSE_PER_AIRMASS = 0.01 * u.mas / u.nm
COMA_HALF_RESPONSE_MAGNITUDE = 16 * u.mag
COMA_ZENITH_RANGE = [0, 40] * u.deg

# WFS gain, noise, and blur.
WFS_SAMPLING = SensingWfsSampling(zero_draws=128, gain_draws=64, block_draws=8, seed=77)
WFS_FIELD_RADIUS = 60 * u.arcsec
WFS_DISPLACEMENT_MAS = 10
WFS_GAIN_AT_BRIGHT_END = 0.85
WFS_GAIN_PER_MAGNITUDE = -0.03
WFS_NOISE_AT_BRIGHT_END_MAS = 2
WFS_NOISE_PER_MAGNITUDE_MAS = 0.5
NOISE_BLUR_WEIGHT = 0.7
NOISE_BLUR_SINGLE_NGS_SCALE = 0.3
NOISE_BLUR_MULTI_NGS_SCALE = 0.6


def main(output_dir: Path) -> None:
    """Generate and save the four synthetic models under the output directory."""
    rng = np.random.default_rng(RANDOM_SEED)
    positions = np.array(list(product(FIELD_AXIS_ARCSEC, repeat=2))) * u.arcsec
    controller = FixedGainIntegrator(
        gain=LO_GAIN, delay=LO_DELAY_FRAMES, frame_rate=LO_FRAME_RATE, dtype=DTYPE
    )
    photometry = CalibrationPhotometry(
        magnitude_band=MAGNITUDE_BAND,
        photons_per_frame_zeropoint=PHOTON_ZEROPOINT,
        dtype=DTYPE,
    )

    reconstructor_noise = _synthetic_reconstructor_noise()
    save_reconstructor_calibration(
        reconstructor_noise, output_dir / "reconstructor" / "model.json"
    )

    covariances = _synthetic_covariances(positions, controller, rng)
    save_covariance_calibration(covariances, output_dir / "covariance" / "model.json")

    coma_response = _synthetic_coma_response(photometry)
    save_coma_calibration(coma_response, output_dir / "coma" / "model.json")

    sensing = _synthetic_sensing_model(positions, controller, photometry, rng)
    save_sensing_calibration(sensing, output_dir / "sensing" / "model.json")


def _synthetic_reconstructor_noise() -> ReconstructorNoiseModel:
    """Assign constant pre-filter noise variance on the example grid."""
    grid_shape = (len(RECONSTRUCTOR_PHOTON_GRID), len(RECONSTRUCTOR_FWHM_GRID_MAS))
    variance = np.full(grid_shape, RECONSTRUCTOR_VARIANCE_MAS2) * u.mas**2
    return ReconstructorNoiseModel(
        photons_per_frame=RECONSTRUCTOR_PHOTON_GRID * u.photon,
        fwhm=RECONSTRUCTOR_FWHM_GRID_MAS * u.mas,
        variance=variance,
        dtype=DTYPE,
    )


def _synthetic_covariances(
    positions: u.Quantity,
    controller: FixedGainIntegrator,
    rng: np.random.Generator,
) -> CovarianceFunctions:
    """Build the filtered covariance functions from generated histories."""
    telemetry = _synthetic_tilt_and_coma(positions, controller, rng)
    settings = CovarianceCalibrationSettings(feedback=COVARIANCE_FEEDBACK_STRENGTHS)
    measurements = measure_covariance_calibration(
        telemetry, controller, settings=settings, dtype=controller.dtype
    )
    return build_covariance_calibration([measurements], dtype=controller.dtype).model


def _synthetic_tilt_and_coma(
    positions: u.Quantity,
    controller: FixedGainIntegrator,
    rng: np.random.Generator,
) -> CovarianceTelemetry:
    """Combine field-wide motion with independent motion at each grid point."""
    shared_shape = (REALIZATIONS, FRAMES_PER_REALIZATION, 1, 2)
    field_shape = (REALIZATIONS, FRAMES_PER_REALIZATION, len(positions), 2)
    shared_motion = rng.normal(size=shared_shape)
    tilt = SHARED_TILT_STD_MAS * shared_motion + LOCAL_TILT_STD_MAS * rng.normal(
        size=field_shape
    )
    coma = SHARED_COMA_STD_NM * shared_motion + LOCAL_COMA_STD_NM * rng.normal(
        size=field_shape
    )
    return CovarianceTelemetry(
        zenith_angle=ZENITH_ANGLE,
        positions=positions,
        frame_rate=controller.frame_rate,
        tilt=tilt * u.mas,
        coma=coma * u.nm,
        dtype=controller.dtype,
    )


def _synthetic_coma_response(photometry: CalibrationPhotometry) -> ComaResponseModel:
    """Set illustrative coma response coefficients for the example."""
    return ComaResponseModel(
        a0=ZENITH_COMA_RESPONSE,
        b=COMA_RESPONSE_PER_AIRMASS,
        m_half=COMA_HALF_RESPONSE_MAGNITUDE,
        magnitude_range=MAGNITUDES * u.mag,
        zenith_range=COMA_ZENITH_RANGE,
        photometry=photometry,
        dtype=photometry.dtype,
    )


def _synthetic_sensing_model(
    positions: u.Quantity,
    controller: FixedGainIntegrator,
    photometry: CalibrationPhotometry,
    rng: np.random.Generator,
) -> SensingModel:
    """Fit the response to generated WFS tilt samples and supply a blur shape."""
    measurements = _synthetic_wfs_measurements(positions, controller, photometry, rng)
    response = build_sensing_response(
        measurements,
        field_radius=WFS_FIELD_RADIUS,
        magnitude_intervals=[MAGNITUDES] * u.mag,
        dtype=controller.dtype,
    )
    blur = NoiseBlurModel(
        mixture_weight=NOISE_BLUR_WEIGHT,
        mixture_single_scale=NOISE_BLUR_SINGLE_NGS_SCALE,
        mixture_multi_scale=NOISE_BLUR_MULTI_NGS_SCALE,
        dtype=controller.dtype,
    )
    return SensingModel(response=response.model, blur=blur)


def _synthetic_wfs_measurements(
    positions: u.Quantity,
    controller: FixedGainIntegrator,
    photometry: CalibrationPhotometry,
    rng: np.random.Generator,
) -> Iterator[SensingCalibrationMeasurements]:
    """Simulate unshifted and paired shifted tilts at each grid point."""
    offsets_mas = WFS_DISPLACEMENT_MAS * np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])
    for position in positions:
        for magnitude in MAGNITUDES:
            magnitude_offset = magnitude - MAGNITUDES[0]
            gain = WFS_GAIN_AT_BRIGHT_END + WFS_GAIN_PER_MAGNITUDE * magnitude_offset
            noise_std_mas = (
                WFS_NOISE_AT_BRIGHT_END_MAS
                + WFS_NOISE_PER_MAGNITUDE_MAS * magnitude_offset
            )
            zero = rng.normal(scale=noise_std_mas, size=(WFS_SAMPLING.zero_draws, 2))
            shared_noise = rng.normal(
                scale=noise_std_mas, size=(WFS_SAMPLING.gain_draws, 2)
            )
            shifted = gain * offsets_mas[:, None, :] + shared_noise[None, :, :]
            samples = SensingWfsMeasurements(
                zero=zero * u.mas,
                shifted=shifted * u.mas,
                dtype=controller.dtype,
            )
            yield SensingCalibrationMeasurements(
                zenith_angle=ZENITH_ANGLE,
                position=position,
                magnitude=magnitude * u.mag,
                samples=samples,
                displacement=WFS_DISPLACEMENT_MAS * u.mas,
                sampling=WFS_SAMPLING,
                controller=controller,
                photometry=photometry,
                dtype=controller.dtype,
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Directory for the four synthetic models.",
    )
    main(parser.parse_args().output_dir)
