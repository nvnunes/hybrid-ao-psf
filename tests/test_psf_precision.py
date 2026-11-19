"""PSF-provider precision, storage and unchanged float32 defaults."""

from __future__ import annotations

import pickle
from dataclasses import replace

import numpy as np
import pytest
from astropy import units as u
from scipy.interpolate import RegularGridInterpolator

from hybrid_ao_psf import (
    PsfPrediction,
    PsfSamples,
    build_psf_interpolator,
    compare_psf_interpolator,
    load_psf_interpolator,
    save_psf_interpolator,
)


def _samples():
    y, x = np.mgrid[-4:5, -4:5]
    image = np.exp(-(x**2 + y**2) / 3.7)
    image *= 0.753246789012345 / image.sum()
    return PsfSamples(
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
    return provider.evaluate(
        zenith_angle=20 * u.deg,
        wavelength=710 * u.nm,
        x=np.asarray(x) * u.arcsec,
        y=np.asarray(y) * u.arcsec,
    )


def test_default_precision_remains_float32(tmp_path):
    samples = _samples()
    model = build_psf_interpolator(samples)
    path = tmp_path / "default.pkl"
    save_psf_interpolator(model, path)
    loaded = load_psf_interpolator(path)
    prediction = _predict(loaded, samples.x.value, samples.y.value)
    assert loaded.psf_grid.dtype == np.float32
    assert prediction.psfs.dtype == np.float32
    np.testing.assert_array_equal(prediction.psfs, samples.psfs.astype(np.float32))


def test_float64_preserves_pixels_flux_interpolation_and_comparison(tmp_path):
    samples = _samples()
    model = build_psf_interpolator(samples, dtype=np.float64)
    path = tmp_path / "double.pkl"
    save_psf_interpolator(model, path)
    loaded = load_psf_interpolator(path)
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
    comparison = compare_psf_interpolator(loaded, original)
    assert comparison.psf_nrms_max == 0


def test_prediction_allows_explicit_double_precision_and_owns_pixels():
    provider = build_psf_interpolator(_samples())
    default = _predict(provider, [0], [0])
    pixels = _samples().psfs[:1].copy()
    prediction = PsfPrediction(
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
        build_psf_interpolator(_samples(), dtype=dtype)
    default = _predict(build_psf_interpolator(_samples()), [0], [0])
    with pytest.raises((TypeError, ValueError), match="float32 or float64"):
        replace(default, dtype=dtype)


def test_loader_rejects_unsupported_stored_precision(tmp_path):
    path = tmp_path / "bad-precision.pkl"
    save_psf_interpolator(
        build_psf_interpolator(_samples()), path
    )
    with path.open("rb") as stream:
        file_contents = pickle.load(stream)
    evaluator = file_contents["model"]["prepared_psf_interpolator"]
    file_contents["model"]["prepared_psf_interpolator"] = RegularGridInterpolator(
        evaluator.grid, evaluator.values.astype(np.float16), bounds_error=True
    )
    with path.open("wb") as stream:
        pickle.dump(file_contents, stream)
    with pytest.raises(ValueError, match="float32 or float64"):
        load_psf_interpolator(path)


def test_float64_preserves_varying_airmass_and_wavelength_axes(tmp_path):
    base = _samples()
    planes = np.stack([base.psfs * a for a in (1, 1.13, 1.29, 1.47)])
    samples = replace(
        base,
        psfs=planes,
        wavelength=[710, 900, 710, 900] * u.nm,
        zenith_angle=[20, 20, 40, 40] * u.deg,
    )
    model = build_psf_interpolator(samples, dtype="float64")
    path = tmp_path / "physical-axes.pkl"
    save_psf_interpolator(model, path)
    loaded = load_psf_interpolator(path)
    for plane, za, wave in zip(
        planes, samples.zenith_angle, samples.wavelength, strict=True
    ):
        prediction = loaded.evaluate(
            zenith_angle=za, wavelength=wave, x=base.x, y=base.y
        )
        np.testing.assert_array_equal(prediction.psfs, plane)
    mid_airmass = np.mean(1 / np.cos(samples.zenith_angle.to_value(u.rad)))
    midpoint = loaded.evaluate(
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
def test_physical_metadata_precision_and_metrics(tmp_path, dtype):
    from ao_core.stats import compute_psf_fwhm

    samples = _samples()
    samples = replace(
        samples,
        pixel_scale=5.123456789012345 * u.mas,
        tel_pupil=np.full((4, 4), 1.00000001234) * u.one,
    )
    provider = build_psf_interpolator(samples, dtype=dtype)
    path = tmp_path / "source.pkl"
    save_psf_interpolator(provider, path)
    provider = load_psf_interpolator(path)
    prediction = _predict(provider, [0.5], [1.25])
    assert prediction.pixel_scale.dtype == dtype
    assert prediction.metadata.tel_pupil.dtype == dtype
    assert provider.pixel_scale_grid.dtype == dtype
    assert provider.grid_precision == np.dtype(dtype)
    np.testing.assert_array_equal(
        provider.tel_pupil.value, samples.tel_pupil.value.astype(dtype)
    )
    metric = compute_psf_fwhm(prediction.psfs, prediction.metadata, dtype=dtype)
    assert metric.dtype == dtype
    other = np.float64 if dtype == np.float32 else np.float32
    converted = load_psf_interpolator(path, dtype=other)
    assert converted.dtype == other
    assert converted.grid_precision == np.dtype(np.float32)
    assert provider.dtype == dtype
    assert not converted.psf_grid.flags.writeable
    assert not np.shares_memory(converted.psf_grid, provider.psf_grid)
    np.testing.assert_array_equal(
        _predict(converted, [0], [0]).psfs,
        _predict(provider, [0], [0]).psfs.astype(other),
    )


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_ngs_metrics_precision_and_loading(tmp_path, dtype):
    from hybrid_ao_psf import (
        PsfSamples,
        build_psf_metric_interpolator_from_psfs,
        load_psf_metric_interpolator,
        save_psf_metric_interpolator,
    )

    s = _samples()
    s = PsfSamples(
        x=s.x,
        y=s.y,
        psfs=s.psfs,
        wavelength=s.wavelength,
        pixel_scale=s.pixel_scale,
        tel_diameter=s.tel_diameter,
        tel_pupil=s.tel_pupil,
        zenith_angle=s.zenith_angle,
    )
    provider = build_psf_metric_interpolator_from_psfs(s, dtype=dtype)
    path = tmp_path / "ngs.pkl"
    save_psf_metric_interpolator(provider, path)
    loaded = load_psf_metric_interpolator(path)
    result = loaded.evaluate(
        zenith_angle=20 * u.deg, x=[0.5] * u.arcsec, y=[1] * u.arcsec
    )
    for value in (result.ee, result.fwhm, result.sr):
        assert value.dtype == dtype
    assert loaded.dtype == dtype
    assert load_psf_metric_interpolator(
        path, dtype=np.float64
    ).grid_precision == np.dtype(dtype)


def test_promoted_science_psfs_retains_physical_endpoints(tmp_path):
    base = _samples()
    samples = replace(
        base,
        psfs=np.stack([base.psfs * a for a in (1, 1.13, 1.29, 1.47)]),
        wavelength=[711, 903, 711, 903] * u.nm,
        zenith_angle=[20.2, 20.2, 40.3, 40.3] * u.deg,
    )
    source = build_psf_interpolator(samples, dtype=np.float32)
    path = tmp_path / "source.pkl"
    save_psf_interpolator(source, path)
    promoted = load_psf_interpolator(path, dtype=np.float64)
    for za, wave in zip(samples.zenith_angle, samples.wavelength, strict=True):
        actual = promoted.evaluate(zenith_angle=za, wavelength=wave, x=base.x, y=base.y)
        expected = source.evaluate(zenith_angle=za, wavelength=wave, x=base.x, y=base.y)
        np.testing.assert_allclose(actual.psfs, expected.psfs, atol=2e-7, rtol=2e-7)
    with pytest.raises(ValueError, match="range"):
        promoted.evaluate(
            zenith_angle=41 * u.deg, wavelength=711 * u.nm, x=base.x, y=base.y
        )
