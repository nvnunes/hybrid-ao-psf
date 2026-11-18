"""Saved PSFs and interpolated PSFs share the simulation input contract."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from astropy import units as u

from hybrid_ao_psf import (
    PsfFileProvider,
    PsfProvider,
    PsfSamples,
    build_psf_interpolator,
    simulate,
)

from .test_engine import _model, _providers, _request, hook_runtime  # noqa: F401


def read_samples(path: Path) -> PsfSamples:
    """Read the independent test cube and its physical coordinates."""
    with np.load(path) as saved:
        return PsfSamples(
            psfs=saved["psfs"],
            x=saved["x"] * u.arcsec,
            y=saved["y"] * u.arcsec,
            wavelength=saved["wavelength"] * u.um,
            zenith_angle=saved["zenith_angle"] * u.deg,
            pixel_scale=saved["pixel_scale"] * u.mas,
            tel_diameter=8 * u.m,
            tel_pupil=np.ones((4, 4)) * u.one,
        )


def write_cube(tmp_path, name="psfs", *, scale=1.0):
    y, x = np.mgrid[-2:3, -2:3]
    psfs = np.stack(
        [np.exp(-((x - dx) ** 2 + y**2)) for dx in (0, 0.2, -0.2, 0.5)]
    )
    psfs *= 0.7 / psfs.sum(axis=(-2, -1), keepdims=True)
    path = tmp_path / f"{name}.npz"
    np.savez(
        path,
        psfs=psfs,
        x=np.array([0, 1, 0, 1], dtype=np.float32),
        y=np.array([0, 0, 1, 1], dtype=np.float32),
        wavelength=np.float32(1.65),
        zenith_angle=np.float32(30),
        pixel_scale=scale,
    )
    return path


@pytest.mark.usefixtures("hook_runtime")
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_file_and_interpolator_agree_through_simulate(tmp_path, dtype):
    path = write_cube(tmp_path)
    samples = read_samples(path)
    saved = PsfFileProvider(files=[path], reader=read_samples)
    interpolated = build_psf_interpolator(samples, dtype=dtype)
    assert isinstance(saved, PsfProvider)
    _, metrics = _providers()
    request = replace(
        _request(),
        dtype=dtype,
        science_positions=[[1, 1], [0, 0]] * u.arcsec,
    )
    model = _model(metrics)
    file_result = simulate(request, science_psfs=saved, lo_model=model)
    interpolation_result = simulate(request, science_psfs=interpolated, lo_model=model)
    np.testing.assert_array_equal(file_result.psfs, interpolation_result.psfs)
    np.testing.assert_array_equal(
        file_result.ctot_angle, interpolation_result.ctot_angle
    )
    assert file_result.metadata.pixel_scale == interpolation_result.metadata.pixel_scale
    assert file_result.psfs.dtype == dtype


@pytest.mark.parametrize(
    "overrides",
    [
        {"x": [0.5] * u.arcsec},
        {"wavelength": 1.7 * u.um},
        {"zenith_angle": 31 * u.deg},
    ],
)
def test_missing_saved_coordinate_is_rejected(tmp_path, overrides):
    saved = PsfFileProvider(files=[write_cube(tmp_path)], reader=read_samples)
    coordinates = {
        "x": [0] * u.arcsec,
        "y": [0] * u.arcsec,
        "wavelength": 1.65 * u.um,
        "zenith_angle": 30 * u.deg,
    }
    with pytest.raises(ValueError, match="No saved PSF"):
        saved.evaluate(**(coordinates | overrides))


def test_multiple_files_can_supply_disjoint_requested_positions(tmp_path):
    paths = [write_cube(tmp_path, name=str(index)) for index in range(2)]

    def reader(path):
        samples = read_samples(path)
        start, stop = (0, 2) if path == paths[0] else (2, 4)
        return replace(
            samples,
            psfs=samples.psfs[start:stop],
            x=samples.x[start:stop],
            y=samples.y[start:stop],
        )

    provider = PsfFileProvider(files=paths, reader=reader)
    result = provider.evaluate(
        x=[1, 0] * u.arcsec,
        y=[1, 0] * u.arcsec,
        wavelength=1.65 * u.um,
        zenith_angle=30 * u.deg,
    )
    np.testing.assert_array_equal(
        result.psfs, read_samples(paths[0]).psfs[[3, 0]].astype(np.float32)
    )


def test_duplicate_saved_match_is_rejected(tmp_path):
    provider = PsfFileProvider(
        files=[write_cube(tmp_path, name="a"), write_cube(tmp_path, name="b")],
        reader=read_samples,
    )
    with pytest.raises(ValueError, match="Multiple saved PSFs"):
        provider.evaluate(
            x=[0] * u.arcsec, y=[0] * u.arcsec,
            wavelength=1.65 * u.um, zenith_angle=30 * u.deg,
        )


def test_reader_owns_files_and_preparation_preserves_flux(tmp_path):
    path = write_cube(tmp_path)
    provider = PsfFileProvider(files=[path], reader=read_samples)
    result = PsfFileProvider(files=[path], reader=read_samples, dtype=np.float64).evaluate(
        x=[0] * u.arcsec, y=[0] * u.arcsec,
        wavelength=1.65 * u.um, zenith_angle=30 * u.deg,
    )
    np.testing.assert_allclose(result.psfs.sum(), 0.7, rtol=1e-7)
    assert result.dtype == np.float64
    assert provider.dtype == np.float32
    assert not result.psfs.flags.writeable
    path.unlink()
    with pytest.raises(FileNotFoundError, match="PSF file"):
        provider.evaluate(
            x=[0] * u.arcsec, y=[0] * u.arcsec,
            wavelength=1.65 * u.um, zenith_angle=30 * u.deg,
        )


def test_repeated_requests_preserve_order_and_image_flux(tmp_path):
    path = write_cube(tmp_path)
    result = PsfFileProvider(files=[path], reader=read_samples).evaluate(
        x=[1, 0, 1] * u.arcsec, y=[1, 0, 1] * u.arcsec,
        wavelength=1.65 * u.um, zenith_angle=30 * u.deg,
    )
    np.testing.assert_array_equal(
        result.psfs, read_samples(path).psfs[[3, 0, 3]].astype(np.float32)
    )


def test_selected_images_require_consistent_sampling(tmp_path):
    paths = [write_cube(tmp_path, name="a"), write_cube(tmp_path, name="b", scale=2)]

    def reader(path):
        samples = read_samples(path)
        index = 0 if path == paths[0] else 3
        return replace(samples, psfs=samples.psfs[index:index + 1],
                       x=samples.x[index:index + 1], y=samples.y[index:index + 1])

    provider = PsfFileProvider(files=paths, reader=reader)
    with pytest.raises(ValueError, match="share image shape, pixel scale, and aperture"):
        provider.evaluate(x=[0, 1] * u.arcsec, y=[0, 1] * u.arcsec,
                         wavelength=1.65 * u.um, zenith_angle=30 * u.deg)


def test_file_reader_can_return_multiple_cubes(tmp_path):
    path = write_cube(tmp_path)

    def reader(path):
        samples = read_samples(path)
        return replace(samples, psfs=np.stack([samples.psfs, 2 * samples.psfs]),
                       wavelength=[1.65, 2.2] * u.um,
                       zenith_angle=[30, 45] * u.deg,
                       pixel_scale=[1, 2] * u.mas)

    provider = PsfFileProvider(files=[path], reader=reader, dtype=np.float64)
    result = provider.evaluate(x=[0] * u.arcsec, y=[0] * u.arcsec,
                              wavelength=2.2 * u.um, zenith_angle=45 * u.deg)
    np.testing.assert_array_equal(result.psfs, 2 * read_samples(path).psfs[[0]])
    assert result.pixel_scale == 2 * u.mas
