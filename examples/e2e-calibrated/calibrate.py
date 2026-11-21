"""Build the four E2E calibration models from saved HDF5 telemetry."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from hybrid_ao_psf import load_telemetry_calibration


def main(ini: Path) -> None:
    """Read the configuration and build and save each calibration in order."""
    calibration = load_telemetry_calibration(ini, dtype=np.float64)

    reconstructor = calibration.build_reconstructor(
        calibration.read_reconstructor_measurements()
    )
    print(f"Saved {calibration.save_reconstructor(reconstructor)}")

    covariance = calibration.build_covariance(
        calibration.measure_covariance(),
        validation_telemetry=calibration.read_covariance_telemetry(),
        reconstructor=reconstructor.model,
    )
    print(f"Saved {calibration.save_covariance(covariance)}")

    coma = calibration.build_coma(calibration.measure_coma())
    print(f"Saved {calibration.save_coma(coma)}")

    response = calibration.build_sensing_response(calibration.read_sensing())
    displacements = calibration.simulate_sensing_noise(
        response,
        calibration.sensing_noise_requests(response),
        covariance=covariance.model,
        reconstructor=reconstructor.model,
    )
    sensing = calibration.build_sensing(response, displacements)
    print(f"Saved {calibration.save_sensing(sensing)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ini", type=Path, help="Example calibration INI.")
    args = parser.parse_args()
    main(args.ini)
