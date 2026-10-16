"""PSF-provider precision, storage and unchanged float32 defaults."""

from __future__ import annotations

import pickle
from dataclasses import replace

import numpy as np
import pytest
from astropy import units as u
from scipy.interpolate import RegularGridInterpolator

from hybrid_ao_psf import (
    ScienceHoPsfPrediction,
    ScienceHoPsfSamples,
    build_science_ho_psf_interpolator,
    load_science_ho_psf_interpolator,
    replay_science_ho_psf_interpolator,
    save_science_ho_psf_interpolator,
)


def _samples():
    y, x = np.mgrid[-4:5, -4:5]
    image = np.exp(-(x**2 + y**2) / 3.7)
    image *= 0.753246789012345 / image.sum()
    return ScienceHoPsfSamples(
        x=[0, 2, 0, 2] * u.arcsec,
        y=[0, 0, 2, 2] * u.arcsec,
        psfs=np.stack([image * a for a in (1, 0.97, 0.91, 0.86)]),
        wavelength=710 * u.nm,
        zenith_angle=20 * u.deg,
        pixel_scale=5 * u.mas,
        tel_diameter=8 * u.m,
        tel_pupil=np.ones((4, 4)) * u.one,
    )


def _predict(provider, x, y):
    return provider.predict(
        zenith_angle=20 * u.deg,
        wavelength=710 * u.nm,
        x=np.asarray(x) * u.arcsec,
        y=np.asarray(y) * u.arcsec,
    )


def test_default_precision_remains_float32(tmp_path):
    samples = _samples()
    model = build_science_ho_psf_interpolator(samples)
    path = tmp_path / "default.pkl"
    save_science_ho_psf_interpolator(model, path)
    loaded = load_science_ho_psf_interpolator(path)
    prediction = _predict(loaded, samples.x.value, samples.y.value)
    assert loaded.psf_grid.dtype == np.float32
    assert prediction.psfs.dtype == np.float32
    np.testing.assert_array_equal(prediction.psfs, samples.psfs.astype(np.float32))


def test_float64_preserves_pixels_flux_interpolation_and_replay(tmp_path):
    samples = _samples()
    model = build_science_ho_psf_interpolator(samples, dtype=np.float64)
    path = tmp_path / "double.pkl"
    save_science_ho_psf_interpolator(model, path)
    loaded = load_science_ho_psf_interpolator(path)
    assert model.dtype == loaded.dtype == np.dtype("float64")
    assert loaded.psf_grid.dtype == np.float64
    assert not loaded.psf_grid.flags.writeable
    samples.psfs[0, 0, 0] += 1e-12
    assert model.psf_grid[0, 0, 0, 0] != samples.psfs[0, 0, 0]
    original = _samples()
    prediction = _predict(loaded, original.x.value, original.y.value)
    assert prediction.dtype == prediction.psfs.dtype == np.dtype("float64")
    assert not prediction.psfs.flags.writeable
    np.testing.assert_array_equal(prediction.psfs, original.psfs)
    # The deliberately non-unit source flux must not be normalized by the provider.
    np.testing.assert_array_equal(prediction.psfs.sum((1, 2)), original.psfs.sum((1, 2)))
    assert not np.array_equal(prediction.psfs, prediction.psfs.astype(np.float32))
    midpoint = _predict(loaded, [1], [1])
    np.testing.assert_allclose(midpoint.psfs[0], original.psfs.mean(0), rtol=1e-15, atol=0)
    replay = replay_science_ho_psf_interpolator(loaded, original)
    assert replay.psf_nrms_max == 0


def test_prediction_allows_explicit_double_precision_and_owns_pixels():
    provider = build_science_ho_psf_interpolator(_samples())
    default = _predict(provider, [0], [0])
    pixels = _samples().psfs[:1].copy()
    prediction = ScienceHoPsfPrediction(
        psfs=pixels,
        pixel_scale=default.pixel_scale,
        metadata=default.metadata,
        dtype="float64",
    )
    expected = pixels.copy()
    pixels.fill(0)
    np.testing.assert_array_equal(prediction.psfs, expected)
    assert prediction.psfs.dtype == np.float64
    assert not prediction.psfs.flags.writeable


@pytest.mark.parametrize("dtype", [np.float16, np.int64, np.complex128, "invalid"])
def test_unsupported_psf_precision_is_rejected(dtype):
    with pytest.raises((TypeError, ValueError), match="float32 or float64"):
        build_science_ho_psf_interpolator(_samples(), dtype=dtype)
    default = _predict(build_science_ho_psf_interpolator(_samples()), [0], [0])
    with pytest.raises((TypeError, ValueError), match="float32 or float64"):
        replace(default, dtype=dtype)


def test_loader_rejects_unsupported_stored_precision(tmp_path):
    path = tmp_path / "bad-precision.pkl"
    save_science_ho_psf_interpolator(build_science_ho_psf_interpolator(_samples()), path)
    with path.open("rb") as stream:
        payload = pickle.load(stream)
    prepared = payload["model"]["prepared_psf_interpolator"]
    payload["model"]["prepared_psf_interpolator"] = RegularGridInterpolator(
        prepared.grid, prepared.values.astype(np.float16), bounds_error=True
    )
    with path.open("wb") as stream:
        pickle.dump(payload, stream)
    with pytest.raises(ValueError, match="float32 or float64"):
        load_science_ho_psf_interpolator(path)


def test_float64_preserves_active_airmass_and_wavelength_axes(tmp_path):
    base = _samples()
    planes = np.stack([base.psfs * a for a in (1, 1.13, 1.29, 1.47)])
    samples = replace(
        base,
        psfs=planes,
        wavelength=[710, 900, 710, 900] * u.nm,
        zenith_angle=[20, 20, 40, 40] * u.deg,
    )
    model = build_science_ho_psf_interpolator(samples, dtype="float64")
    path = tmp_path / "physical-axes.pkl"
    save_science_ho_psf_interpolator(model, path)
    loaded = load_science_ho_psf_interpolator(path)
    for plane, za, wave in zip(planes, samples.zenith_angle, samples.wavelength, strict=True):
        prediction = loaded.predict(zenith_angle=za, wavelength=wave, x=base.x, y=base.y)
        np.testing.assert_array_equal(prediction.psfs, plane)
    mid_airmass = np.mean(1 / np.cos(samples.zenith_angle.to_value(u.rad)))
    midpoint = loaded.predict(
        zenith_angle=np.arccos(1 / mid_airmass) * u.rad,
        wavelength=805 * u.nm,
        x=[1] * u.arcsec,
        y=[1] * u.arcsec,
    )
    assert midpoint.psfs.dtype == np.float64
    np.testing.assert_allclose(midpoint.psfs[0], planes.mean((0, 1)), rtol=1e-14, atol=0)
