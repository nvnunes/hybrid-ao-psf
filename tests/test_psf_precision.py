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
    np.testing.assert_array_equal(
        prediction.psfs.sum((1, 2)), original.psfs.sum((1, 2))
    )
    assert not np.array_equal(prediction.psfs, prediction.psfs.astype(np.float32))
    midpoint = _predict(loaded, [1], [1])
    np.testing.assert_allclose(
        midpoint.psfs[0], original.psfs.mean(0), rtol=1e-15, atol=0
    )
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
    save_science_ho_psf_interpolator(
        build_science_ho_psf_interpolator(_samples()), path
    )
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
    for plane, za, wave in zip(
        planes, samples.zenith_angle, samples.wavelength, strict=True
    ):
        prediction = loaded.predict(
            zenith_angle=za, wavelength=wave, x=base.x, y=base.y
        )
        np.testing.assert_array_equal(prediction.psfs, plane)
    mid_airmass = np.mean(1 / np.cos(samples.zenith_angle.to_value(u.rad)))
    midpoint = loaded.predict(
        zenith_angle=np.arccos(1 / mid_airmass) * u.rad,
        wavelength=805 * u.nm,
        x=[1] * u.arcsec,
        y=[1] * u.arcsec,
    )
    assert midpoint.psfs.dtype == np.float64
    np.testing.assert_allclose(
        midpoint.psfs[0], planes.mean((0, 1)), rtol=1e-14, atol=0
    )


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_physical_metadata_runtime_preparation_and_metrics(tmp_path, dtype):
    from ao_stats import compute_psf_fwhm

    samples = _samples()
    samples = replace(
        samples,
        pixel_scale=5.123456789012345 * u.mas,
        tel_pupil=np.full((4, 4), 1.00000001234) * u.one,
    )
    provider = build_science_ho_psf_interpolator(samples, dtype=dtype)
    path = tmp_path / "source.pkl"
    save_science_ho_psf_interpolator(provider, path)
    provider = load_science_ho_psf_interpolator(path)
    prediction = _predict(provider, [0.5], [1.25])
    assert prediction.pixel_scale.dtype == dtype
    assert prediction.metadata.tel_pupil.dtype == dtype
    assert provider.pixel_scale_grid.dtype == dtype
    assert provider.builder["precision"] == np.dtype(dtype).name
    np.testing.assert_array_equal(
        provider.tel_pupil.value, samples.tel_pupil.value.astype(dtype)
    )
    metric = compute_psf_fwhm(prediction.psfs, prediction.metadata, dtype=dtype)
    assert metric.dtype == dtype
    other = np.float64 if dtype == np.float32 else np.float32
    prepared = provider.prepare(dtype=other)
    assert prepared.dtype == other
    assert prepared.builder["precision"] == np.dtype(dtype).name
    assert provider.dtype == dtype
    assert not prepared.psf_grid.flags.writeable
    assert not np.shares_memory(prepared.psf_grid, provider.psf_grid)
    np.testing.assert_array_equal(
        _predict(prepared, [0], [0]).psfs,
        _predict(provider, [0], [0]).psfs.astype(other),
    )


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_ngs_provider_precision_and_loading(tmp_path, dtype):
    from hybrid_ao_psf import (
        NgsHoPsfSamples,
        build_ngs_ho_metric_interpolator_from_psfs,
        load_ngs_ho_metric_interpolator,
        save_ngs_ho_metric_interpolator,
    )

    s = _samples()
    s = NgsHoPsfSamples(
        x=s.x,
        y=s.y,
        psfs=s.psfs,
        wavelength=s.wavelength,
        pixel_scale=s.pixel_scale,
        tel_diameter=s.tel_diameter,
        tel_pupil=s.tel_pupil,
        zenith_angle=s.zenith_angle,
    )
    provider = build_ngs_ho_metric_interpolator_from_psfs(s, dtype=dtype)
    path = tmp_path / "ngs.pkl"
    save_ngs_ho_metric_interpolator(provider, path)
    loaded = load_ngs_ho_metric_interpolator(path)
    result = loaded.predict(
        zenith_angle=20 * u.deg, x=[0.5] * u.arcsec, y=[1] * u.arcsec
    )
    for value in (result.ee, result.fwhm, result.sr):
        assert value.dtype == dtype
    assert loaded.dtype == dtype
    assert loaded.prepare(dtype=np.float64).builder["precision"] == np.dtype(dtype).name


def test_promoted_science_provider_retains_physical_endpoints():
    base = _samples()
    samples = replace(
        base,
        psfs=np.stack([base.psfs * a for a in (1, 1.13, 1.29, 1.47)]),
        wavelength=[711, 903, 711, 903] * u.nm,
        zenith_angle=[20.2, 20.2, 40.3, 40.3] * u.deg,
    )
    source = build_science_ho_psf_interpolator(samples, dtype=np.float32)
    promoted = source.prepare(dtype=np.float64)
    for za, wave in zip(samples.zenith_angle, samples.wavelength, strict=True):
        actual = promoted.predict(zenith_angle=za, wavelength=wave, x=base.x, y=base.y)
        expected = source.predict(zenith_angle=za, wavelength=wave, x=base.x, y=base.y)
        np.testing.assert_allclose(actual.psfs, expected.psfs, atol=2e-7, rtol=2e-7)
    with pytest.raises(ValueError, match="support"):
        promoted.predict(
            zenith_angle=41 * u.deg, wavelength=711 * u.nm, x=base.x, y=base.y
        )
