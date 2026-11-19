from __future__ import annotations

import pickle
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from astropy import units as u

from hybrid_ao_psf import PsfMetricProvider, PsfProvider
from hybrid_ao_psf.interpolation import (
    PsfMetricInterpolator,
    PsfMetricSamples,
    PsfSamples,
    _core,
    build_psf_interpolator,
    build_psf_metric_interpolator,
    build_psf_metric_interpolator_from_psfs,
    compare_psf_interpolator,
    compare_psf_metric_interpolator,
    load_psf_interpolator,
    load_psf_metric_interpolator,
    measure_psf_metrics,
    psf,
    psf_metrics,
    save_psf_interpolator,
    save_psf_metric_interpolator,
    zenith_angle_to_airmass,
)
from hybrid_ao_psf.interpolation._core import grid_field_values, rectangular_field_axes


def test_loaded_provider_contracts_are_read_only() -> None:
    science = build_psf_interpolator(_science_samples())
    ngs = build_psf_metric_interpolator(_ngs_metric_samples())

    assert isinstance(science, PsfProvider)
    assert isinstance(ngs, PsfMetricProvider)
    assert not hasattr(ngs, "model")
    assert not hasattr(ngs, "interpolation_data")
    assert "_interpolation_data" not in repr(ngs)
    assert not science.psf_grid.flags.writeable
    assert not science.x.value.flags.writeable
    with pytest.raises(TypeError):
        ngs._interpolation_data["changed"] = True


@pytest.mark.parametrize("stage", ["built", "loaded_float64", "loaded"])
def test_provider_storage_cannot_be_changed_via_public_views(stage, tmp_path):
    images = build_psf_interpolator(_science_samples())
    metrics = build_psf_metric_interpolator(_ngs_metric_samples())
    if stage in ("loaded", "loaded_float64"):
        image_path = tmp_path / "images.pkl"
        metric_path = tmp_path / "metrics.pkl"
        save_psf_interpolator(images, image_path)
        save_psf_metric_interpolator(metrics, metric_path)
        dtype = np.float64 if stage == "loaded_float64" else None
        images = load_psf_interpolator(image_path, dtype=dtype)
        metrics = load_psf_metric_interpolator(metric_path, dtype=dtype)

    coordinates = {"zenith_angle": 20 * u.deg, "x": [0] * u.arcsec, "y": [0] * u.arcsec}
    expected_images = images.evaluate(wavelength=1 * u.um, **coordinates).psfs
    expected_ee = metrics.evaluate(**coordinates).ee
    exposed = (
        images.psf_grid,
        images.pixel_scale_grid.value,
        metrics.x.value,
        metrics._interpolation_data["metric_grids"]["ee"].value,
    )
    for array in exposed:
        while isinstance(array, np.ndarray):
            with pytest.raises(ValueError, match="WRITEABLE"):
                array.flags.writeable = True
            array = array.base

    images.pixel_scale_grid._set_unit(u.arcsec)
    metrics._interpolation_data["metric_grids"]["ee"]._set_unit(u.mas)
    assert images.pixel_scale_grid.unit == u.mas
    assert metrics._interpolation_data["metric_grids"]["ee"].unit == u.one
    np.testing.assert_array_equal(
        images.evaluate(wavelength=1 * u.um, **coordinates).psfs, expected_images
    )
    np.testing.assert_array_equal(metrics.evaluate(**coordinates).ee, expected_ee)


def test_same_psf_samples_build_image_and_metric_interpolators() -> None:
    samples = _ngs_psf_samples()

    images = build_psf_interpolator(samples)
    metrics = measure_psf_metrics(samples)
    metric_interpolator = build_psf_metric_interpolator(metrics)

    coordinates = {
        "zenith_angle": samples.zenith_angle[0],
        "x": samples.x,
        "y": samples.y,
    }
    psf_prediction = images.evaluate(wavelength=samples.wavelength[0], **coordinates)
    metric_prediction = metric_interpolator.evaluate(**coordinates)
    np.testing.assert_array_equal(psf_prediction.psfs, samples.psfs[0])
    np.testing.assert_array_equal(metric_prediction.fwhm.value, metrics.fwhm[0].value)


def test_rbf_provider_protects_fitted_scipy_state() -> None:
    ngs = build_psf_metric_interpolator(
        _scattered_ngs_metric_samples(),
    )
    model = ngs._interpolation_data["evaluators"]["ee"]
    prediction = ngs.evaluate(
        zenith_angle=20.0 * u.deg,
        x=np.asarray([-1.0]) * u.arcsec,
        y=np.asarray([-1.0]) * u.arcsec,
    )

    assert not hasattr(model, "_coeffs")
    with pytest.raises(AttributeError, match="read-only"):
        model.changed = True
    np.testing.assert_allclose(prediction.ee.to_value(u.one), [0.2])


def test_failed_ngs_file_overwrite_preserves_existing_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "ngs.pkl"
    path.write_bytes(b"existing file")
    interpolator = build_psf_metric_interpolator(_ngs_metric_samples())

    def fail_dump(*_: object, **__: object) -> None:
        raise RuntimeError("write failed")

    monkeypatch.setattr(_core.pickle, "dump", fail_dump)
    with pytest.raises(RuntimeError, match="write failed"):
        save_psf_metric_interpolator(interpolator, path, overwrite=True)

    assert path.read_bytes() == b"existing file"
    assert list(tmp_path.glob(".ngs.pkl.*.tmp")) == []


def test_failed_science_file_overwrite_preserves_existing_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "science.pkl"
    path.write_bytes(b"existing file")
    interpolator = build_psf_interpolator(_science_samples())

    def fail_write(*_: object, **__: object) -> None:
        raise RuntimeError("write failed")

    monkeypatch.setattr(_core.pickle, "dump", fail_write)
    with pytest.raises(RuntimeError, match="write failed"):
        save_psf_interpolator(interpolator, path, overwrite=True)

    assert path.read_bytes() == b"existing file"
    assert list(tmp_path.glob(".science.pkl.*.tmp")) == []


def test_psf_interpolator_round_trips_and_compares_samples(tmp_path) -> None:
    interpolator = build_psf_interpolator(_science_samples())
    path = tmp_path / "science.pkl"

    save_psf_interpolator(interpolator, path)
    loaded = load_psf_interpolator(path)
    comparison = compare_psf_interpolator(loaded, _science_samples())
    prediction = loaded.evaluate(
        zenith_angle=20.0 * u.deg,
        wavelength=1.0 * u.um,
        x=_science_samples().x,
        y=_science_samples().y,
    )
    expected_flux = np.sum(_science_samples().psfs[0], axis=(-2, -1))

    assert loaded.coordinate_order == ("airmass", "wavelength", "y", "x")
    np.testing.assert_allclose(
        np.sum(prediction.psfs, axis=(-2, -1)), expected_flux, rtol=1.0e-6
    )
    assert not np.allclose(expected_flux, 1.0)
    assert comparison.num_cubes == _science_samples().psfs.shape[0]
    assert comparison.psf_nrms_max == pytest.approx(0.0, abs=1.0e-5)
    assert comparison.pixel_scale_max_abs_error.to_value(u.mas) == pytest.approx(0.0)
    assert comparison.metric_max_abs_error["fwhm"].to_value(u.mas) == pytest.approx(
        0.0, abs=1.0e-5
    )
    assert comparison.metric_max_abs_error["ee"].to_value(u.one) == pytest.approx(
        0.0, abs=1.0e-5
    )


def test_psf_pickle_rejects_old_file_version(tmp_path: Path) -> None:
    path = tmp_path / "science.pkl"
    save_psf_interpolator(build_psf_interpolator(_science_samples()), path)
    with path.open("rb") as handle:
        file_contents = pickle.load(handle)
    file_contents["version"] = 1
    with path.open("wb") as handle:
        pickle.dump(file_contents, handle)

    with pytest.raises(ValueError, match="Unsupported file version"):
        load_psf_interpolator(path)


def test_science_pickle_does_not_reference_scipy_private_array_adapter(
    tmp_path: Path,
) -> None:
    path = tmp_path / "science.pkl"
    save_psf_interpolator(build_psf_interpolator(_science_samples()), path)
    with path.open("rb") as handle:
        file_contents = pickle.load(handle)

    assert file_contents["model"]["prepared_psf_interpolator"]._asarray is np.asarray


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("stage", ["built", "loaded", "converted"])
def test_psf_evaluator_is_built_once_and_reused(tmp_path, monkeypatch, dtype, stage):
    samples = _science_samples()
    interpolator = build_psf_interpolator(samples, dtype=dtype)
    if stage != "built":
        path = tmp_path / "science.pkl"
        save_psf_interpolator(interpolator, path)
        if stage == "converted":
            dtype = np.float64 if dtype == np.float32 else np.float32
        interpolator = load_psf_interpolator(path, dtype=dtype)

    coordinates = {
        "zenith_angle": 20.0 * u.deg,
        "wavelength": 1.0 * u.um,
        "x": samples.x,
        "y": samples.y,
    }
    # A fresh evaluator uses the same stored coordinate precision without caching.
    build_evaluator = psf._build_evaluator
    fresh_evaluator = build_evaluator(interpolator)
    physical_values = psf._physical_coordinate_values(
        interpolator,
        zenith_angle=coordinates["zenith_angle"],
        wavelength=coordinates["wavelength"],
    )
    field_points = np.column_stack(
        [samples.x.to_value(u.arcsec), samples.y.to_value(u.arcsec)]
    )
    points = psf._evaluation_points(interpolator, physical_values, field_points)
    expected = np.asarray(fresh_evaluator(points), dtype=dtype).reshape(
        samples.x.size, *interpolator.psf_shape
    )
    calls = 0

    def observed_build(provider):
        nonlocal calls
        calls += 1
        return build_evaluator(provider)

    monkeypatch.setattr(psf, "_build_evaluator", observed_build)
    first = interpolator.evaluate(**coordinates)
    evaluator = interpolator._get_evaluator()
    second = interpolator.evaluate(**coordinates)

    assert calls == (0 if stage == "loaded" else 1)
    assert interpolator._get_evaluator() is evaluator
    assert first.psfs.dtype == np.dtype(dtype)
    np.testing.assert_array_equal(first.psfs, expected)
    np.testing.assert_array_equal(first.psfs, second.psfs)


def test_psf_interpolator_uses_rectangular_grid() -> None:
    interpolator = build_psf_interpolator(_science_samples())

    assert interpolator.psf_grid.shape == (2, 2, 2, 2, 9, 9)
    np.testing.assert_allclose(interpolator.x, np.asarray([-1.0, 1.0]) * u.arcsec)
    np.testing.assert_allclose(interpolator.y, np.asarray([-1.0, 1.0]) * u.arcsec)


def test_science_field_coordinates_preserve_directional_psf_pixels() -> None:
    x = np.array([1.0, -1.0, 1.0, -1.0])
    y = np.array([-1.0, -1.0, 1.0, 1.0])
    psfs = np.zeros((4, 9, 9), dtype=np.float32)
    for index, (field_x, field_y) in enumerate(zip(x, y, strict=True)):
        psfs[index, 4, 4] = 1.0
        psfs[index, 4 + int(field_y), 4 + int(field_x)] = 0.4 + 0.1 * index
        psfs[index, 1, 6] = 0.05
    samples = PsfSamples(
        x=x * u.arcsec,
        y=y * u.arcsec,
        psfs=psfs,
        wavelength=1.65 * u.um,
        pixel_scale=4.0 * u.mas,
        tel_diameter=8.0 * u.m,
        tel_pupil=np.ones((4, 4), dtype=np.float32) * u.one,
    )

    prediction = build_psf_interpolator(samples).evaluate(
        zenith_angle=20.0 * u.deg,
        wavelength=1.65 * u.um,
        x=np.array([1.0, -1.0]) * u.arcsec,
        y=np.array([-1.0, 1.0]) * u.arcsec,
    )

    np.testing.assert_array_equal(prediction.psfs, psfs[[0, 3]])
    assert prediction.psfs[0, 3, 5] > prediction.psfs[0, 5, 3]
    assert prediction.psfs[1, 5, 3] > prediction.psfs[1, 3, 5]


def test_rectangular_field_helpers_fill_values_in_y_x_order() -> None:
    x = np.asarray([1.0, -1.0, 1.0, -1.0])
    y = np.asarray([1.0, -1.0, -1.0, 1.0])
    values = np.asarray([4.0, 1.0, 2.0, 3.0])

    x_axis, y_axis = rectangular_field_axes(
        x * u.arcsec, y * u.arcsec, label="test samples"
    )
    grid = grid_field_values(
        x * u.arcsec,
        y * u.arcsec,
        values,
        x_axis,
        y_axis,
        label="test samples",
        dtype=float,
    )

    np.testing.assert_allclose(x_axis, np.asarray([-1.0, 1.0]))
    np.testing.assert_allclose(y_axis, np.asarray([-1.0, 1.0]))
    np.testing.assert_allclose(grid, np.asarray([[1.0, 2.0], [3.0, 4.0]]))


def test_psf_interpolator_interpolates_wavelength_zenith_and_pixel_scale() -> None:
    samples = _science_samples()
    interpolator = build_psf_interpolator(samples)
    airmass_axis = zenith_angle_to_airmass(np.asarray([20.0, 30.0]) * u.deg)
    requested_airmass = float(np.mean(airmass_axis))
    requested_zenith = float(np.rad2deg(np.arccos(1.0 / requested_airmass)))

    prediction = interpolator.evaluate(
        zenith_angle=requested_zenith * u.deg,
        wavelength=1.5 * u.um,
        x=np.asarray([0.0]) * u.arcsec,
        y=np.asarray([0.0]) * u.arcsec,
    )

    assert prediction.psfs.shape == (1, 9, 9)
    assert float(np.sum(prediction.psfs)) > 1.0
    assert prediction.pixel_scale.to_value(u.mas) == pytest.approx(6.375)
    assert prediction.metadata.pixel_scale.to_value(u.mas) == pytest.approx(6.375)
    with pytest.raises(ValueError, match="x"):
        interpolator.evaluate(
            zenith_angle=requested_zenith * u.deg,
            wavelength=1.5 * u.um,
            x=np.asarray([2.0]) * u.arcsec,
            y=np.asarray([0.0]) * u.arcsec,
        )


def test_psf_field_only_file_omits_physical_coordinates(tmp_path) -> None:
    source = _science_samples()
    samples = PsfSamples(
        x=source.x,
        y=source.y,
        psfs=source.psfs[0],
        wavelength=1.0 * u.um,
        pixel_scale=4.0 * u.mas,
        tel_diameter=source.tel_diameter,
        tel_pupil=source.tel_pupil,
    )

    interpolator = build_psf_interpolator(samples)
    path = tmp_path / "science-field.pkl"
    save_psf_interpolator(interpolator, path)
    loaded = load_psf_interpolator(path)
    comparison = compare_psf_interpolator(loaded, samples)
    prediction = loaded.evaluate(
        x=np.asarray([0.0]) * u.arcsec,
        y=np.asarray([0.0]) * u.arcsec,
    )

    assert loaded.coordinate_order == ("y", "x")
    assert loaded.psf_grid.shape == (2, 2, 9, 9)
    assert prediction.metadata.wavelength.to_value(u.um) == pytest.approx(1.0)
    assert prediction.pixel_scale.to_value(u.mas) == pytest.approx(4.0)
    assert comparison.num_cubes == 1
    assert comparison.psf_nrms_max == pytest.approx(0.0, abs=1.0e-5)
    with pytest.raises(ValueError, match="wavelength"):
        loaded.evaluate(
            wavelength=2.0 * u.um,
            x=np.asarray([0.0]) * u.arcsec,
            y=np.asarray([0.0]) * u.arcsec,
        )


def test_psf_supports_single_physical_axes() -> None:
    source = _science_samples()
    wavelength_samples = PsfSamples(
        zenith_angle=20.0 * u.deg,
        wavelength=source.wavelength[:2],
        x=source.x,
        y=source.y,
        psfs=source.psfs[:2],
        pixel_scale=source.pixel_scale[:2],
        tel_diameter=source.tel_diameter,
        tel_pupil=source.tel_pupil,
    )
    wavelength_file = build_psf_interpolator(wavelength_samples)

    assert wavelength_file.coordinate_order == ("wavelength", "y", "x")
    prediction = wavelength_file.evaluate(
        wavelength=1.5 * u.um,
        x=np.asarray([0.0]) * u.arcsec,
        y=np.asarray([0.0]) * u.arcsec,
    )
    assert prediction.pixel_scale.to_value(u.mas) == pytest.approx(6.0)

    airmass_samples = PsfSamples(
        zenith_angle=np.asarray([20.0, 30.0]) * u.deg,
        wavelength=1.0 * u.um,
        x=source.x,
        y=source.y,
        psfs=source.psfs[[0, 2]],
        pixel_scale=source.pixel_scale[[0, 2]],
        tel_diameter=source.tel_diameter,
        tel_pupil=source.tel_pupil,
    )
    airmass_file = build_psf_interpolator(airmass_samples)

    assert airmass_file.coordinate_order == ("airmass", "y", "x")
    with pytest.raises(ValueError, match="zenith_angle is required"):
        airmass_file.evaluate(
            x=np.asarray([0.0]) * u.arcsec,
            y=np.asarray([0.0]) * u.arcsec,
        )


def test_psf_requires_only_varying_physical_coordinates() -> None:
    interpolator = build_psf_interpolator(_science_samples())

    with pytest.raises(ValueError, match="zenith_angle is required"):
        interpolator.evaluate(
            wavelength=1.0 * u.um,
            x=np.asarray([0.0]) * u.arcsec,
            y=np.asarray([0.0]) * u.arcsec,
        )
    with pytest.raises(ValueError, match="wavelength is required"):
        interpolator.evaluate(
            zenith_angle=20.0 * u.deg,
            x=np.asarray([0.0]) * u.arcsec,
            y=np.asarray([0.0]) * u.arcsec,
        )


def test_psf_evaluation_rejects_out_of_range_coordinates() -> None:
    interpolator = build_psf_interpolator(_science_samples())

    with pytest.raises(ValueError, match="wavelength"):
        interpolator.evaluate(
            x=[0.0] * u.arcsec,
            y=[0.0] * u.arcsec,
            zenith_angle=20.0 * u.deg,
            wavelength=3.0 * u.um,
        )
    with pytest.raises(ValueError, match="airmass"):
        interpolator.evaluate(
            x=[0.0] * u.arcsec,
            y=[0.0] * u.arcsec,
            zenith_angle=40.0 * u.deg,
            wavelength=1.5 * u.um,
        )


@pytest.mark.parametrize(
    ("zenith_angle", "wavelength", "message"),
    [
        (
            20.0 * u.deg,
            1.1 * u.um,
            "wavelength=1.1 does not match this interpolator's fixed wavelength=1.0",
        ),
        (
            21.0 * u.deg,
            1.0 * u.um,
            "zenith_angle=21.0 does not match this interpolator's fixed zenith_angle=20.0",
        ),
    ],
)
def test_psf_fixed_coordinate_errors(
    zenith_angle: u.Quantity, wavelength: u.Quantity, message: str
) -> None:
    samples = _science_samples()
    interpolator = build_psf_interpolator(
        PsfSamples(
            zenith_angle=20.0 * u.deg,
            wavelength=1.0 * u.um,
            x=samples.x,
            y=samples.y,
            psfs=samples.psfs[0],
            pixel_scale=samples.pixel_scale[0],
            tel_diameter=samples.tel_diameter,
            tel_pupil=samples.tel_pupil,
        )
    )

    with pytest.raises(ValueError, match=message):
        interpolator.evaluate(
            x=[0.0] * u.arcsec,
            y=[0.0] * u.arcsec,
            zenith_angle=zenith_angle,
            wavelength=wavelength,
        )


def test_psf_interpolator_rejects_malformed_grids() -> None:
    samples = _science_samples()

    with pytest.raises(ValueError, match="complete varying physical-coordinate grid"):
        build_psf_interpolator(
            PsfSamples(
                zenith_angle=samples.zenith_angle[:-1],
                wavelength=samples.wavelength[:-1],
                x=samples.x,
                y=samples.y,
                psfs=samples.psfs[:-1],
                pixel_scale=samples.pixel_scale[:-1],
                tel_diameter=samples.tel_diameter,
                tel_pupil=samples.tel_pupil,
            )
        )

    with pytest.raises(ValueError, match="pixel_scale"):
        build_psf_interpolator(
            PsfSamples(
                zenith_angle=samples.zenith_angle,
                wavelength=samples.wavelength,
                x=samples.x,
                y=samples.y,
                psfs=samples.psfs,
                pixel_scale=np.array([4.0, -1.0, 4.5, 9.0]) * u.mas,
                tel_diameter=samples.tel_diameter,
                tel_pupil=samples.tel_pupil,
            )
        )

    zero_flux = np.array(samples.psfs, copy=True)
    zero_flux[0, 0] = 0.0
    with pytest.raises(ValueError, match="strictly positive per-PSF total flux"):
        build_psf_interpolator(
            PsfSamples(
                zenith_angle=samples.zenith_angle,
                wavelength=samples.wavelength,
                x=samples.x,
                y=samples.y,
                psfs=zero_flux,
                pixel_scale=samples.pixel_scale,
                tel_diameter=samples.tel_diameter,
                tel_pupil=samples.tel_pupil,
            )
        )

    with pytest.raises(ValueError, match="complete rectangular"):
        build_psf_interpolator(
            PsfSamples(
                zenith_angle=samples.zenith_angle,
                wavelength=samples.wavelength,
                x=samples.x[:-1],
                y=samples.y[:-1],
                psfs=samples.psfs[:, :-1],
                pixel_scale=samples.pixel_scale,
                tel_diameter=samples.tel_diameter,
                tel_pupil=samples.tel_pupil,
            )
        )


def test_image_grid_precision_round_trip_and_required_field(tmp_path) -> None:
    provider = build_psf_interpolator(_science_samples())
    path = tmp_path / "interpolator.pkl"
    save_psf_interpolator(provider, path)
    provider = load_psf_interpolator(path, dtype=np.float64)
    save_psf_interpolator(provider, path, overwrite=True)
    restored = load_psf_interpolator(path)
    assert restored.dtype == np.dtype(np.float64)
    assert restored.grid_precision == np.dtype(np.float32)
    with path.open("rb") as handle:
        file_contents = pickle.load(handle)
    assert file_contents["grid_precision"] == "float32"
    assert "builder" not in file_contents
    assert "provenance" not in file_contents["metadata"]
    assert "meta" not in file_contents["metadata"]
    del file_contents["grid_precision"]
    with path.open("wb") as handle:
        pickle.dump(file_contents, handle)
    with pytest.raises(TypeError, match="grid_precision string"):
        load_psf_interpolator(path)


def test_image_interpolator_rejects_version_two(tmp_path) -> None:
    path = tmp_path / "interpolator.pkl"
    save_psf_interpolator(build_psf_interpolator(_science_samples()), path)
    with path.open("rb") as handle:
        file_contents = pickle.load(handle)
    file_contents["version"] = 2
    with path.open("wb") as handle:
        pickle.dump(file_contents, handle)
    with pytest.raises(ValueError, match="Unsupported file version: 2"):
        load_psf_interpolator(path)


def test_psf_metric_interpolator_round_trips_and_compares_samples(tmp_path) -> None:
    samples = _scattered_ngs_metric_samples()
    interpolator = build_psf_metric_interpolator(
        samples,
        dtype=np.float64,
    )
    path = tmp_path / "ngs.pkl"

    save_psf_metric_interpolator(interpolator, path)
    loaded = load_psf_metric_interpolator(path)
    comparison = compare_psf_metric_interpolator(loaded, samples)
    prediction = loaded.evaluate(
        zenith_angle=np.repeat(samples.zenith_angle, samples.x.size),
        x=np.tile(samples.x, samples.zenith_angle.size),
        y=np.tile(samples.y, samples.zenith_angle.size),
    )

    assert loaded.interpolation_type == "rbf"
    assert set(loaded.metric_names) == {"ee", "fwhm", "sr"}
    assert comparison.metric_max_abs_error["ee"].to_value(u.one) == pytest.approx(
        0.0, abs=1.0e-8
    )
    assert comparison.metric_max_abs_error["fwhm"].to_value(u.mas) == pytest.approx(
        0.0, abs=1.0e-8
    )
    assert comparison.metric_max_abs_error["sr"].to_value(u.one) == pytest.approx(
        0.0, abs=1.0e-8
    )
    assert prediction.sr is not None


def test_psf_metric_interpolator_uses_regular_grid_by_default() -> None:
    interpolator = build_psf_metric_interpolator(_ngs_metric_samples())

    assert interpolator.interpolation_type == "regular_grid"
    assert set(interpolator._interpolation_data["metric_grids"]) == {"ee", "fwhm", "sr"}


def test_psf_metric_regular_grid_round_trips_compares_and_interpolates(
    tmp_path,
) -> None:
    samples = _ngs_metric_samples()
    interpolator = build_psf_metric_interpolator(samples, dtype=np.float64)
    path = tmp_path / "ngs-grid.pkl"

    save_psf_metric_interpolator(interpolator, path)
    loaded = load_psf_metric_interpolator(path)
    comparison = compare_psf_metric_interpolator(loaded, samples)
    prediction = loaded.evaluate(
        zenith_angle=20.0 * u.deg,
        x=np.asarray([0.0]) * u.arcsec,
        y=np.asarray([0.0]) * u.arcsec,
    )

    assert loaded.interpolation_type == "regular_grid"
    assert comparison.metric_max_abs_error["ee"].to_value(u.one) == pytest.approx(
        0.0, abs=1.0e-12
    )
    assert comparison.metric_max_abs_error["fwhm"].to_value(u.mas) == pytest.approx(
        0.0, abs=1.0e-12
    )
    assert comparison.metric_max_abs_error["sr"].to_value(u.one) == pytest.approx(
        0.0, abs=1.0e-12
    )
    np.testing.assert_allclose(prediction.ee, np.asarray([0.26]) * u.one)
    np.testing.assert_allclose(prediction.fwhm, np.asarray([83.0]) * u.mas)
    np.testing.assert_allclose(prediction.sr, np.asarray([0.115]) * u.one)


def test_psf_metric_field_only_regular_grid_and_rbf_omit_zenith() -> None:
    source = _ngs_metric_samples()
    samples = PsfMetricSamples(
        zenith_angle=20.0 * u.deg,
        x=source.x,
        y=source.y,
        ee=source.ee[0],
        fwhm=source.fwhm[0],
        sr=source.sr[0],
    )

    regular = build_psf_metric_interpolator(samples)
    scattered = _scattered_ngs_metric_samples()
    rbf = build_psf_metric_interpolator(
        replace(
            scattered,
            zenith_angle=20 * u.deg,
            ee=scattered.ee[0],
            fwhm=scattered.fwhm[0],
            sr=scattered.sr[0],
        ),
        dtype=np.float64,
    )

    assert regular.coordinate_order == ("y", "x")
    assert rbf.coordinate_order == ("x", "y")
    assert "wavelength" not in regular.coordinate_order
    assert "wavelength" not in rbf.coordinate_order
    regular_prediction = regular.evaluate(
        x=np.asarray([0.0]) * u.arcsec,
        y=np.asarray([0.0]) * u.arcsec,
    )
    rbf_prediction = rbf.evaluate(
        x=np.asarray([-1.0]) * u.arcsec,
        y=np.asarray([-1.0]) * u.arcsec,
    )
    np.testing.assert_allclose(regular_prediction.ee, np.asarray([0.26]) * u.one)
    np.testing.assert_allclose(
        rbf_prediction.ee, np.asarray([0.20]) * u.one, atol=1.0e-8
    )
    with pytest.raises(ValueError, match="interpolator's fixed zenith_angle=20.0"):
        regular.evaluate(
            zenith_angle=30.0 * u.deg,
            x=np.asarray([0.0]) * u.arcsec,
            y=np.asarray([0.0]) * u.arcsec,
        )


def test_psf_metric_varying_airmass_requires_zenith() -> None:
    regular = build_psf_metric_interpolator(_ngs_metric_samples())
    rbf = build_psf_metric_interpolator(
        _scattered_ngs_metric_samples(),
    )

    assert regular.coordinate_order == ("airmass", "y", "x")
    assert rbf.coordinate_order == ("airmass", "x", "y")
    with pytest.raises(ValueError, match="zenith_angle is required"):
        regular.evaluate(
            x=np.asarray([0.0]) * u.arcsec,
            y=np.asarray([0.0]) * u.arcsec,
        )
    with pytest.raises(ValueError, match="zenith_angle is required"):
        rbf.evaluate(
            x=np.asarray([0.0]) * u.arcsec,
            y=np.asarray([0.0]) * u.arcsec,
        )


def test_psf_metric_interpolator_broadcasts_scalar_zenith() -> None:
    samples = _ngs_metric_samples()
    interpolator = build_psf_metric_interpolator(
        samples,
        dtype=np.float64,
    )

    prediction = interpolator.evaluate(
        zenith_angle=20.0 * u.deg,
        x=np.asarray([-1.0, 1.0]) * u.arcsec,
        y=np.asarray([-1.0, 1.0]) * u.arcsec,
    )

    assert prediction.ee.shape == (2,)
    assert prediction.fwhm.shape == (2,)
    assert prediction.sr.shape == (2,)


def test_psf_metric_evaluation_rejects_unavailable_coordinates() -> None:
    interpolator = build_psf_metric_interpolator(
        _scattered_ngs_metric_samples(),
    )

    interpolator.evaluate(
        zenith_angle=20.0 * u.deg,
        x=np.asarray([0.0]) * u.arcsec,
        y=np.asarray([0.0]) * u.arcsec,
    )
    with pytest.raises(ValueError) as above:
        interpolator.evaluate(
            zenith_angle=40.0 * u.deg,
            x=np.asarray([0.0]) * u.arcsec,
            y=np.asarray([0.0]) * u.arcsec,
        )
    assert str(above.value) == (
        "zenith_angle=40.0 deg is outside this interpolator's "
        "zenith_angle range [20.0, 30.0] deg."
    )
    with pytest.raises(ValueError) as below:
        interpolator.evaluate(
            zenith_angle=10.0 * u.deg,
            x=np.asarray([0.0]) * u.arcsec,
            y=np.asarray([0.0]) * u.arcsec,
        )
    assert str(below.value) == (
        "zenith_angle=10.0 deg is outside this interpolator's "
        "zenith_angle range [20.0, 30.0] deg."
    )
    with pytest.raises(ValueError, match="field"):
        interpolator.evaluate(
            zenith_angle=20.0 * u.deg,
            x=np.asarray([2.0]) * u.arcsec,
            y=np.asarray([0.0]) * u.arcsec,
        )


def test_psf_metric_regular_grid_evaluation_rejects_unavailable_coordinates() -> None:
    interpolator = build_psf_metric_interpolator(_ngs_metric_samples())

    interpolator.evaluate(
        zenith_angle=20.0 * u.deg,
        x=np.asarray([0.0]) * u.arcsec,
        y=np.asarray([0.0]) * u.arcsec,
    )
    with pytest.raises(ValueError, match="zenith_angle=40.0 deg is outside"):
        interpolator.evaluate(
            zenith_angle=40.0 * u.deg,
            x=np.asarray([0.0]) * u.arcsec,
            y=np.asarray([0.0]) * u.arcsec,
        )
    with pytest.raises(ValueError, match="x"):
        interpolator.evaluate(
            zenith_angle=20.0 * u.deg,
            x=np.asarray([2.0]) * u.arcsec,
            y=np.asarray([0.0]) * u.arcsec,
        )


def test_psf_metric_interpolator_rejects_bad_metric_values() -> None:
    samples = _ngs_metric_samples()

    with pytest.raises(ValueError, match="ee"):
        build_psf_metric_interpolator(
            PsfMetricSamples(
                zenith_angle=samples.zenith_angle,
                x=samples.x,
                y=samples.y,
                ee=np.full(samples.ee.shape, 1.2) * u.one,
                fwhm=samples.fwhm,
                sr=samples.sr,
            )
        )
    with pytest.raises(ValueError, match="fwhm"):
        build_psf_metric_interpolator(
            PsfMetricSamples(
                zenith_angle=samples.zenith_angle,
                x=samples.x,
                y=samples.y,
                ee=samples.ee,
                fwhm=np.full(samples.fwhm.shape, -1.0) * u.mas,
                sr=samples.sr,
            )
        )
    with pytest.raises(ValueError, match="sr"):
        build_psf_metric_interpolator(
            PsfMetricSamples(
                zenith_angle=samples.zenith_angle,
                x=samples.x,
                y=samples.y,
                ee=samples.ee,
                fwhm=samples.fwhm,
                sr=np.full(samples.sr.shape, -1.0) * u.one,
            )
        )


def test_psf_metric_samples_require_metric_quantities() -> None:
    samples = _ngs_metric_samples()

    with pytest.raises(TypeError, match="ee must be an Astropy Quantity"):
        build_psf_metric_interpolator(
            PsfMetricSamples(
                zenith_angle=samples.zenith_angle,
                x=samples.x,
                y=samples.y,
                ee=np.asarray(samples.ee.value),
                fwhm=samples.fwhm,
                sr=samples.sr,
            )
        )


def test_psf_metric_selection_accepts_scattered_points_and_rejects_duplicates() -> None:
    samples = _ngs_metric_samples()
    scattered = replace(
        samples,
        x=samples.x[:-1],
        y=samples.y[:-1],
        ee=samples.ee[:, :-1],
        fwhm=samples.fwhm[:, :-1],
        sr=samples.sr[:, :-1],
    )
    assert build_psf_metric_interpolator(scattered).interpolation_type == "rbf"

    with pytest.raises(ValueError, match="Duplicate PSF metric samples field point"):
        build_psf_metric_interpolator(
            replace(
                samples,
                x=[-1.0, -1.0, 1.0, 1.0] * u.arcsec,
                y=[-1.0, -1.0, 1.0, 1.0] * u.arcsec,
            )
        )


def test_psf_metric_interpolator_rejects_interpolation_data_metric_mismatch() -> None:
    interpolator = build_psf_metric_interpolator(
        _scattered_ngs_metric_samples(),
    )
    interpolation_data = dict(interpolator._interpolation_data)
    evaluators = dict(interpolation_data["evaluators"])
    del evaluators["sr"]
    interpolation_data["evaluators"] = evaluators
    bad = PsfMetricInterpolator(
        coordinate_order=interpolator.coordinate_order,
        zenith_angle_axis=interpolator.zenith_angle_axis,
        airmass_axis=interpolator.airmass_axis,
        x=interpolator.x,
        y=interpolator.y,
        metric_names=interpolator.metric_names,
        interpolation_type=interpolator.interpolation_type,
        _interpolation_data=interpolation_data,
    )

    with pytest.raises(
        ValueError, match="interpolation data does not match metric_names"
    ):
        psf_metrics._validate_interpolator(bad)


def test_ngs_ho_psfs_feed_metric_samples_then_metric_interpolator() -> None:
    psf_samples = _ngs_psf_samples()
    metric_samples = measure_psf_metrics(psf_samples)
    direct = build_psf_metric_interpolator(metric_samples)
    from_psfs = build_psf_metric_interpolator_from_psfs(
        psf_samples,
    )

    assert metric_samples.sr is not None
    assert metric_samples.ee.shape == (2, 4)
    assert metric_samples.fwhm.shape == (2, 4)
    assert metric_samples.sr.shape == (2, 4)
    assert set(direct.metric_names) == {"ee", "fwhm", "sr"}
    assert set(from_psfs.metric_names) == {"ee", "fwhm", "sr"}
    assert np.all(np.isfinite(metric_samples.ee))
    assert np.all(np.isfinite(metric_samples.fwhm))
    assert np.all(np.isfinite(metric_samples.sr))


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_ngs_metric_normalization_is_preserved_only_when_requested(
    tmp_path, dtype
) -> None:
    samples = _ngs_psf_samples()
    scaled = replace(samples, psfs=0.4 * samples.psfs)
    default = measure_psf_metrics(scaled, dtype=dtype)
    explicit_default = measure_psf_metrics(
        scaled, preserve_normalization=False, dtype=dtype
    )
    preserved = measure_psf_metrics(scaled, preserve_normalization=True, dtype=dtype)
    for name in ("ee", "sr", "fwhm"):
        np.testing.assert_array_equal(
            getattr(default, name), getattr(explicit_default, name)
        )
        factor = 1.0 if name == "fwhm" else 0.4
        np.testing.assert_allclose(
            getattr(preserved, name).value,
            factor * getattr(default, name).value,
            rtol=2e-6,
        )
    file = build_psf_metric_interpolator_from_psfs(
        scaled,
        preserve_normalization=True,
        dtype=dtype,
    )
    prediction = file.evaluate(zenith_angle=20.0 * u.deg, x=samples.x, y=samples.y)
    for name in ("ee", "sr", "fwhm"):
        np.testing.assert_allclose(
            getattr(prediction, name).value,
            getattr(preserved, name).value[0],
            rtol=2e-6,
        )
    output = tmp_path / "full-flux.pkl"
    save_psf_metric_interpolator(file, output)
    loaded = load_psf_metric_interpolator(output)
    coordinates = {
        "zenith_angle": 25 * u.deg,
        "x": np.array([-1.0, 0.0, 1.0]) * u.arcsec,
        "y": np.array([-1.0, 0.0, 1.0]) * u.arcsec,
    }
    before, after = file.evaluate(**coordinates), loaded.evaluate(**coordinates)
    legacy = build_psf_metric_interpolator_from_psfs(scaled, dtype=dtype).evaluate(
        **coordinates
    )
    for name in ("ee", "sr", "fwhm"):
        np.testing.assert_array_equal(getattr(before, name), getattr(after, name))
        factor = 1.0 if name == "fwhm" else 0.4
        np.testing.assert_allclose(
            getattr(after, name).value, factor * getattr(legacy, name).value, rtol=2e-6
        )


def test_ngs_ho_psf_samples_require_full_psf_metadata() -> None:
    samples = _ngs_psf_samples()

    with pytest.raises(ValueError, match="tel_pupil"):
        measure_psf_metrics(
            PsfSamples(
                zenith_angle=samples.zenith_angle,
                x=samples.x,
                y=samples.y,
                psfs=samples.psfs,
                wavelength=samples.wavelength,
                pixel_scale=samples.pixel_scale,
                tel_diameter=samples.tel_diameter,
                tel_pupil=np.ones(4) * u.dimensionless_unscaled,
            )
        )


def _science_samples() -> PsfSamples:
    zenith = np.asarray([20.0, 20.0, 30.0, 30.0])
    wavelength = np.asarray([1.0, 2.0, 1.0, 2.0])
    x = np.asarray([-1.0, 1.0, -1.0, 1.0])
    y = np.asarray([-1.0, -1.0, 1.0, 1.0])
    psfs = np.empty((4, 4, 9, 9), dtype=np.float32)
    for plane in range(4):
        for point, (px, py) in enumerate(zip(x, y, strict=True)):
            psfs[plane, point] = (2.0 + 0.5 * plane + 0.25 * point) * _gaussian_psf(
                9, sigma=1.0 + 0.05 * plane, dx=0.1 * px, dy=0.1 * py
            )
    return PsfSamples(
        zenith_angle=zenith * u.deg,
        wavelength=wavelength * u.um,
        x=x * u.arcsec,
        y=y * u.arcsec,
        psfs=psfs,
        pixel_scale=np.asarray([4.0, 8.0, 4.5, 9.0]) * u.mas,
        tel_diameter=30.0 * u.m,
        tel_pupil=np.ones((4, 4), dtype=np.float32) * u.dimensionless_unscaled,
    )


def _ngs_metric_samples() -> PsfMetricSamples:
    zenith = np.asarray([20.0, 30.0])
    x = np.asarray([-1.0, 1.0, -1.0, 1.0])
    y = np.asarray([-1.0, -1.0, 1.0, 1.0])
    ee = np.asarray(
        [
            [0.20, 0.24, 0.28, 0.32],
            [0.30, 0.34, 0.38, 0.42],
        ],
        dtype=float,
    )
    fwhm = np.asarray(
        [
            [80.0, 82.0, 84.0, 86.0],
            [90.0, 92.0, 94.0, 96.0],
        ],
        dtype=float,
    )
    sr = np.asarray(
        [
            [0.10, 0.11, 0.12, 0.13],
            [0.15, 0.16, 0.17, 0.18],
        ],
        dtype=float,
    )
    return PsfMetricSamples(
        zenith_angle=zenith * u.deg,
        x=x * u.arcsec,
        y=y * u.arcsec,
        ee=ee * u.dimensionless_unscaled,
        fwhm=fwhm * u.mas,
        sr=sr * u.dimensionless_unscaled,
    )


def _ngs_psf_samples() -> PsfSamples:
    zenith = np.asarray([20.0, 30.0])
    x = np.asarray([-1.0, 1.0, -1.0, 1.0])
    y = np.asarray([-1.0, -1.0, 1.0, 1.0])
    psfs = np.empty((2, 4, 21, 21), dtype=np.float32)
    for plane in range(2):
        for point, (px, py) in enumerate(zip(x, y, strict=True)):
            psfs[plane, point] = _gaussian_psf(
                21,
                sigma=1.2 + 0.05 * plane + 0.03 * point,
                dx=0.1 * px,
                dy=0.1 * py,
            )
    return PsfSamples(
        zenith_angle=zenith * u.deg,
        x=x * u.arcsec,
        y=y * u.arcsec,
        psfs=psfs,
        wavelength=np.asarray([1.65, 1.65]) * u.um,
        pixel_scale=np.asarray([5.0, 5.0]) * u.mas,
        tel_diameter=30.0 * u.m,
        tel_pupil=np.ones((8, 8), dtype=np.float32) * u.dimensionless_unscaled,
    )


def _gaussian_psf(
    size: int, *, sigma: float, dx: float = 0.0, dy: float = 0.0
) -> np.ndarray:
    coords = np.arange(size, dtype=float) - (size - 1) / 2.0
    xx, yy = np.meshgrid(coords - dx, coords - dy)
    psf = np.exp(-0.5 * (xx**2 + yy**2) / sigma**2)
    return (psf / psf.sum()).astype(np.float32)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_scattered_metrics_match_fixed_rbf_method_and_roundtrip(tmp_path, dtype):
    from scipy.interpolate import RBFInterpolator

    samples = _scattered_ngs_metric_samples()
    # Nonlinear samples distinguish the smoothing choice from an exact polynomial fit.
    fwhm = samples.fwhm.copy()
    fwhm[:, -1] += 3 * u.mas
    samples = replace(samples, fwhm=fwhm)
    fitted = build_psf_metric_interpolator(samples, dtype=dtype)
    assert fitted.interpolation_type == "rbf"
    path = tmp_path / "scattered.pkl"
    save_psf_metric_interpolator(fitted, path)
    loaded = load_psf_metric_interpolator(path)
    coordinates = np.column_stack(
        [
            np.repeat(
                1
                / np.cos(
                    np.deg2rad(np.asarray(samples.zenith_angle.value, dtype=dtype))
                ),
                samples.x.size,
            ),
            np.tile(
                np.asarray(samples.x.value, dtype=dtype), samples.zenith_angle.size
            ),
            np.tile(
                np.asarray(samples.y.value, dtype=dtype), samples.zenith_angle.size
            ),
        ]
    ).astype(dtype)
    mean, scale = coordinates.mean(axis=0), coordinates.std(axis=0)
    reference = RBFInterpolator(
        (coordinates - mean) / scale,
        np.asarray(samples.fwhm.value, dtype=dtype).ravel(),
        kernel="thin_plate_spline",
        smoothing=0.05,
        degree=1,
    )
    requested = np.asarray(
        [[1 / np.cos(np.deg2rad(dtype(25))), 0.35, -0.2]], dtype=dtype
    )
    expected = reference((requested - mean) / scale)
    for model in (fitted, loaded):
        actual = model.evaluate(
            zenith_angle=25 * u.deg, x=[0.35] * u.arcsec, y=[-0.2] * u.arcsec
        )
        np.testing.assert_allclose(actual.fwhm.value, expected, rtol=2e-6)
    with path.open("rb") as handle:
        file_contents = pickle.load(handle)
    assert file_contents["version"] == 4
    assert file_contents["interpolation_type"] == "rbf"
    assert "interpolation_config" not in file_contents
    assert "config" not in file_contents["model"]
    assert set(file_contents["model"]["evaluators"]) == {"ee", "fwhm", "sr"}
    assert "models" not in file_contents["model"]
    assert {"coordinate_mean", "coordinate_scale", "varying_axes"} <= file_contents[
        "model"
    ].keys()
    assert (
        not {"coord_mean", "coord_scale", "active_dims"} & file_contents["model"].keys()
    )

    file_contents["model"]["models"] = file_contents["model"].pop("evaluators")
    with path.open("wb") as handle:
        pickle.dump(file_contents, handle)
    with pytest.raises(ValueError, match="missing interpolation data metrics"):
        load_psf_metric_interpolator(path)


def _scattered_ngs_metric_samples() -> PsfMetricSamples:
    samples = _ngs_metric_samples()
    return replace(
        samples,
        x=np.append(samples.x.value, 0) * u.arcsec,
        y=np.append(samples.y.value, 0) * u.arcsec,
        **{
            name: np.column_stack(
                [
                    getattr(samples, name).value,
                    getattr(samples, name).value.mean(axis=1),
                ]
            )
            * getattr(samples, name).unit
            for name in ("ee", "fwhm", "sr")
        },
    )


@pytest.mark.parametrize("scattered", [False, True])
def test_psf_metric_selection_rejects_duplicate_zenith_samples(scattered):
    samples = _scattered_ngs_metric_samples() if scattered else _ngs_metric_samples()
    with pytest.raises(ValueError, match="duplicate zenith-angle samples"):
        build_psf_metric_interpolator(replace(samples, zenith_angle=[20, 20] * u.deg))


def test_zero_smoothing_preserves_scattered_metric_at_sample_point(tmp_path):
    samples = _scattered_ngs_metric_samples()
    ee = samples.ee.copy()
    ee[0, -1] += 0.03 * u.one
    samples = replace(samples, ee=ee)
    exact = build_psf_metric_interpolator(samples, dtype=np.float64, rbf_smoothing=0.0)
    default = build_psf_metric_interpolator(samples, dtype=np.float64)
    query = {"zenith_angle": samples.zenith_angle[0], "x": samples.x[-1], "y": samples.y[-1]}

    path = tmp_path / "exact.pkl"
    save_psf_metric_interpolator(exact, path)
    restored = load_psf_metric_interpolator(path)
    np.testing.assert_allclose(restored.evaluate(**query).ee, ee[0, -1], atol=1e-12)
    assert abs(default.evaluate(**query).ee - ee[0, -1]) > 1e-5 * u.one


@pytest.mark.parametrize("reverse", [False, True])
def test_rectangular_metric_selection_is_independent_of_sample_order(reverse):
    samples = _ngs_metric_samples()
    order = [2, 0, 3, 1] if reverse else [0, 1, 2, 3]
    shuffled = replace(
        samples,
        x=samples.x[order],
        y=samples.y[order],
        **{name: getattr(samples, name)[:, order] for name in ("ee", "fwhm", "sr")},
    )
    model = build_psf_metric_interpolator(shuffled, dtype=np.float64)
    assert model.interpolation_type == "regular_grid"
    actual = model.evaluate(
        zenith_angle=20 * u.deg, x=[0.2] * u.arcsec, y=[-0.3] * u.arcsec
    )
    np.testing.assert_allclose(actual.fwhm.value, [82.6], rtol=1e-12)
