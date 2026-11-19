"""Compare Hybrid predictions from saved PSFs and an interpolator at saved coordinates."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np
from psf_fits import read_psf_samples

from hybrid_ao_psf import (
    DiagnosticsLevel,
    MastselModel,
    PsfFileProvider,
    load_configuration,
    load_psf_interpolator,
    load_psf_metric_interpolator,
    simulate,
)


def main(
    configuration: Path,
    science_dir: Path,
    science_interpolator: Path,
    ngs_metrics: Path,
) -> None:
    """Use the same request and LO model with both sources of science PSFs."""
    files = sorted(science_dir.glob("*.fits"))
    if not files:
        raise ValueError(f"{science_dir} contains no FITS PSF cubes.")
    samples = read_psf_samples(files[0])
    indices = [0, len(samples.x) // 2]
    configuration = load_configuration(configuration)
    request = replace(
        configuration.request,
        science_positions=np.column_stack((samples.x[indices], samples.y[indices])),
        wavelength=samples.wavelength,
        diagnostics_level=DiagnosticsLevel.DEBUG,
        zenith_angle=samples.zenith_angle,
    )
    saved_psfs = PsfFileProvider(files=files, reader=read_psf_samples)
    interpolated_psfs = load_psf_interpolator(science_interpolator)
    model = MastselModel(
        mastsel_ini=configuration.mastsel_ini,
        ngs_metrics=load_psf_metric_interpolator(ngs_metrics),
        non_psd_policy=configuration.non_psd_policy,
    )

    query = {
        "wavelength": request.wavelength,
        "zenith_angle": request.zenith_angle,
        "x": request.science_positions[:, 0],
        "y": request.science_positions[:, 1],
    }
    saved = saved_psfs.evaluate(**query)
    interpolated = interpolated_psfs.evaluate(**query)
    np.testing.assert_array_equal(saved.psfs, interpolated.psfs)
    assert saved.pixel_scale == interpolated.pixel_scale
    np.testing.assert_array_equal(
        saved.metadata.tel_pupil, interpolated.metadata.tel_pupil
    )

    file_result = simulate(request, science_psfs=saved_psfs, lo_model=model)
    interpolated_result = simulate(
        request, science_psfs=interpolated_psfs, lo_model=model
    )
    np.testing.assert_array_equal(file_result.psfs, interpolated_result.psfs)
    np.testing.assert_array_equal(
        file_result.diagnostics.covariance.ctot, interpolated_result.diagnostics.covariance.ctot
    )
    print("Saved images, pixel scale, pupil, final PSFs, and LO covariance agree.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configuration", required=True, type=Path)
    parser.add_argument("--science-psfs", required=True, type=Path)
    parser.add_argument("--science-interpolator", required=True, type=Path)
    parser.add_argument("--ngs-metrics", required=True, type=Path)
    args = parser.parse_args()
    main(
        args.configuration,
        args.science_psfs,
        args.science_interpolator,
        args.ngs_metrics,
    )
