from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pytest
from astropy import units as u

from hybrid_ao_psf import NgsMetricProvider, SciencePsfProvider
from hybrid_ao_psf.interpolation import (
    NgsHoMetricInterpolator,
    NgsHoMetricSamples,
    NgsHoPsfSamples,
    RbfInterpolationConfig,
    RegularGridInterpolationConfig,
    ScienceHoPsfSamples,
    _core,
    build_ngs_ho_metric_interpolator,
    build_ngs_ho_metric_interpolator_from_psfs,
    build_ngs_ho_metric_samples_from_psfs,
    build_science_ho_psf_interpolator,
    evaluate_ngs_ho_metric_interpolator,
    evaluate_science_ho_psf_interpolator,
    load_ngs_ho_metric_interpolator,
    load_science_ho_psf_interpolator,
    replay_ngs_ho_metric_interpolator,
    replay_science_ho_psf_interpolator,
    save_ngs_ho_metric_interpolator,
    save_science_ho_psf_interpolator,
    science_ho_psf,
    validate_ngs_ho_metric_interpolator,
    validate_ngs_ho_metric_query,
    validate_science_ho_psf_query,
    zenith_angle_to_airmass,
)
from hybrid_ao_psf.interpolation._core import grid_field_values, rectangular_field_axes


def test_loaded_provider_contracts_are_read_only() -> None:
    science = build_science_ho_psf_interpolator(_science_samples())
    ngs = build_ngs_ho_metric_interpolator(_ngs_metric_samples())

    assert isinstance(science, SciencePsfProvider)
    assert isinstance(ngs, NgsMetricProvider)
    assert not science.psf_grid.flags.writeable
    assert not science.x.value.flags.writeable
    with pytest.raises(TypeError):
        science.builder["changed"] = True
    with pytest.raises(TypeError):
        ngs.model["changed"] = True


def test_rbf_provider_hides_and_freezes_fitted_scipy_state() -> None:
    ngs = build_ngs_ho_metric_interpolator(
        _ngs_metric_samples(),
        interpolation_config=RbfInterpolationConfig(smoothing=0.0),
    )
    model = ngs.model["models"]["ee"]
    prediction = ngs.predict(
        zenith_angle=20.0 * u.deg,
        x=np.asarray([-1.0]) * u.arcsec,
        y=np.asarray([-1.0]) * u.arcsec,
    )

    assert not hasattr(model, "_coeffs")
    with pytest.raises(AttributeError, match="immutable"):
        model.changed = True
    np.testing.assert_allclose(prediction.ee.to_value(u.one), [0.2])


def test_failed_ngs_artifact_overwrite_preserves_existing_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "ngs.pkl"
    path.write_bytes(b"existing artifact")
    interpolator = build_ngs_ho_metric_interpolator(_ngs_metric_samples())

    def fail_dump(*_: object, **__: object) -> None:
        raise RuntimeError("write failed")

    monkeypatch.setattr(_core.pickle, "dump", fail_dump)
    with pytest.raises(RuntimeError, match="write failed"):
        save_ngs_ho_metric_interpolator(interpolator, path, overwrite=True)

    assert path.read_bytes() == b"existing artifact"
    assert list(tmp_path.glob(".ngs.pkl.*.tmp")) == []


def test_failed_science_artifact_overwrite_preserves_existing_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "science.pkl"
    path.write_bytes(b"existing artifact")
    interpolator = build_science_ho_psf_interpolator(_science_samples())

    def fail_write(*_: object, **__: object) -> None:
        raise RuntimeError("write failed")

    monkeypatch.setattr(_core.pickle, "dump", fail_write)
    with pytest.raises(RuntimeError, match="write failed"):
        save_science_ho_psf_interpolator(interpolator, path, overwrite=True)

    assert path.read_bytes() == b"existing artifact"
    assert list(tmp_path.glob(".science.pkl.*.tmp")) == []


def test_science_ho_psf_interpolator_round_trips_and_replays_sources(tmp_path) -> None:
    interpolator = build_science_ho_psf_interpolator(_science_samples())
    path = tmp_path / "science.pkl"

    save_science_ho_psf_interpolator(interpolator, path)
    loaded = load_science_ho_psf_interpolator(path)
    replay = replay_science_ho_psf_interpolator(loaded, _science_samples())
    prediction = evaluate_science_ho_psf_interpolator(
        loaded,
        zenith_angle=20.0 * u.deg,
        wavelength=1.0 * u.um,
        x=_science_samples().x,
        y=_science_samples().y,
    )
    expected_flux = np.sum(_science_samples().psfs[0], axis=(-2, -1))

    assert loaded.builder["interpolation_method"] == "regular_grid_linear_airmass_wavelength_y_x"
    assert loaded.coordinate_order == ("airmass", "wavelength", "y", "x")
    assert loaded.meta["norm_correction"].to_value(u.one) == pytest.approx(0.75)
    np.testing.assert_allclose(
        loaded.meta["plane_correction"],
        np.asarray([[1.0, 2.0], [3.0, 4.0]]) * u.one,
    )
    assert prediction.meta["norm_correction"].to_value(u.one) == pytest.approx(0.75)
    assert prediction.meta["plane_correction"].to_value(u.one) == pytest.approx(1.0)
    np.testing.assert_allclose(np.sum(prediction.psfs, axis=(-2, -1)), expected_flux, rtol=1.0e-6)
    assert not np.allclose(expected_flux, 1.0)
    assert replay.psf_nrms_max == pytest.approx(0.0, abs=1.0e-5)
    assert replay.pixel_scale_abs_max.to_value(u.mas) == pytest.approx(0.0)
    assert replay.metric_max_abs["fwhm"].to_value(u.mas) == pytest.approx(0.0, abs=1.0e-5)
    assert replay.metric_max_abs["ee"].to_value(u.one) == pytest.approx(0.0, abs=1.0e-5)


def test_science_ho_psf_pickle_rejects_old_artifact_version(tmp_path: Path) -> None:
    path = tmp_path / "science.pkl"
    save_science_ho_psf_interpolator(build_science_ho_psf_interpolator(_science_samples()), path)
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    payload["version"] = 1
    with path.open("wb") as handle:
        pickle.dump(payload, handle)

    with pytest.raises(ValueError, match="Unsupported artifact version"):
        load_science_ho_psf_interpolator(path)


def test_science_pickle_does_not_reference_scipy_private_array_adapter(tmp_path: Path) -> None:
    path = tmp_path / "science.pkl"
    save_science_ho_psf_interpolator(build_science_ho_psf_interpolator(_science_samples()), path)
    with path.open("rb") as handle:
        payload = pickle.load(handle)

    assert payload["model"]["prepared_psf_interpolator"]._asarray is np.asarray


def test_loaded_science_provider_reuses_persisted_runtime_for_repeated_queries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "science.pkl"
    save_science_ho_psf_interpolator(build_science_ho_psf_interpolator(_science_samples()), path)
    interpolator = load_science_ho_psf_interpolator(path)
    prepare = science_ho_psf._prepare_science_ho_psf_runtime_interpolator
    calls = 0

    def observed_prepare(provider):
        nonlocal calls
        calls += 1
        return prepare(provider)

    monkeypatch.setattr(science_ho_psf, "_prepare_science_ho_psf_runtime_interpolator", observed_prepare)
    query = {
        "zenith_angle": 20.0 * u.deg,
        "wavelength": 1.0 * u.um,
        "x": _science_samples().x,
        "y": _science_samples().y,
    }
    reference = science_ho_psf._evaluate_science_ho_psf_runtime_interpolator(
        prepare(interpolator), **query
    )

    first = interpolator.predict(**query)
    second = interpolator.predict(**query)
    via_function = evaluate_science_ho_psf_interpolator(interpolator, **query)

    assert calls == 0
    np.testing.assert_array_equal(first.psfs, reference.psfs)
    np.testing.assert_array_equal(first.psfs, second.psfs)
    np.testing.assert_array_equal(first.psfs, via_function.psfs)


def test_science_evaluator_rejects_wrong_provider_type() -> None:
    with pytest.raises(TypeError, match="ScienceHoPsfInterpolator"):
        evaluate_science_ho_psf_interpolator(
            object(),
            zenith_angle=20.0 * u.deg,
            wavelength=1.0 * u.um,
            x=np.asarray([0.0]) * u.arcsec,
            y=np.asarray([0.0]) * u.arcsec,
        )


def test_science_ho_psf_interpolator_uses_rectangular_grid() -> None:
    interpolator = build_science_ho_psf_interpolator(_science_samples())

    assert interpolator.psf_grid.shape == (2, 2, 2, 2, 9, 9)
    np.testing.assert_allclose(interpolator.x, np.asarray([-1.0, 1.0]) * u.arcsec)
    np.testing.assert_allclose(interpolator.y, np.asarray([-1.0, 1.0]) * u.arcsec)


def test_science_field_queries_preserve_directional_psf_pixels() -> None:
    x = np.array([1.0, -1.0, 1.0, -1.0])
    y = np.array([-1.0, -1.0, 1.0, 1.0])
    psfs = np.zeros((4, 9, 9), dtype=np.float32)
    for index, (field_x, field_y) in enumerate(zip(x, y, strict=True)):
        psfs[index, 4, 4] = 1.0
        psfs[index, 4 + int(field_y), 4 + int(field_x)] = 0.4 + 0.1 * index
        psfs[index, 1, 6] = 0.05
    samples = ScienceHoPsfSamples(
        x=x * u.arcsec,
        y=y * u.arcsec,
        psfs=psfs,
        wavelength=1.65 * u.um,
        pixel_scale=4.0 * u.mas,
        tel_diameter=8.0 * u.m,
        tel_pupil=np.ones((4, 4), dtype=np.float32) * u.one,
    )

    prediction = build_science_ho_psf_interpolator(samples).predict(
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


def test_science_ho_psf_interpolator_interpolates_wavelength_zenith_and_pixel_scale() -> None:
    samples = _science_samples()
    interpolator = build_science_ho_psf_interpolator(samples)
    airmass_axis = zenith_angle_to_airmass(np.asarray([20.0, 30.0]) * u.deg)
    query_airmass = float(np.mean(airmass_axis))
    query_zenith = float(np.rad2deg(np.arccos(1.0 / query_airmass)))

    prediction = evaluate_science_ho_psf_interpolator(
        interpolator,
        zenith_angle=query_zenith * u.deg,
        wavelength=1.5 * u.um,
        x=np.asarray([0.0]) * u.arcsec,
        y=np.asarray([0.0]) * u.arcsec,
    )

    assert prediction.psfs.shape == (1, 9, 9)
    assert float(np.sum(prediction.psfs)) > 1.0
    assert prediction.pixel_scale.to_value(u.mas) == pytest.approx(6.375)
    assert prediction.metadata.pixel_scale.to_value(u.mas) == pytest.approx(6.375)
    assert prediction.meta["norm_correction"].to_value(u.one) == pytest.approx(0.75)
    assert prediction.meta["plane_correction"].to_value(u.one) == pytest.approx(2.5)
    with pytest.raises(ValueError, match="x"):
        evaluate_science_ho_psf_interpolator(
            interpolator,
            zenith_angle=query_zenith * u.deg,
            wavelength=1.5 * u.um,
            x=np.asarray([2.0]) * u.arcsec,
            y=np.asarray([0.0]) * u.arcsec,
        )


def test_science_ho_psf_field_only_artifact_omits_physical_queries(tmp_path) -> None:
    source = _science_samples()
    samples = ScienceHoPsfSamples(
        x=source.x,
        y=source.y,
        psfs=source.psfs[0],
        wavelength=1.0 * u.um,
        pixel_scale=4.0 * u.mas,
        tel_diameter=source.tel_diameter,
        tel_pupil=source.tel_pupil,
        provenance=("field-only",),
    )

    interpolator = build_science_ho_psf_interpolator(samples)
    path = tmp_path / "science-field.pkl"
    save_science_ho_psf_interpolator(interpolator, path)
    loaded = load_science_ho_psf_interpolator(path)
    replay = replay_science_ho_psf_interpolator(loaded, samples)
    prediction = evaluate_science_ho_psf_interpolator(
        loaded,
        x=np.asarray([0.0]) * u.arcsec,
        y=np.asarray([0.0]) * u.arcsec,
    )

    assert loaded.coordinate_order == ("y", "x")
    assert loaded.psf_grid.shape == (2, 2, 9, 9)
    assert prediction.metadata.wavelength.to_value(u.um) == pytest.approx(1.0)
    assert prediction.pixel_scale.to_value(u.mas) == pytest.approx(4.0)
    assert replay.psf_nrms_max == pytest.approx(0.0, abs=1.0e-5)
    with pytest.raises(ValueError, match="wavelength"):
        evaluate_science_ho_psf_interpolator(
            loaded,
            wavelength=2.0 * u.um,
            x=np.asarray([0.0]) * u.arcsec,
            y=np.asarray([0.0]) * u.arcsec,
        )


def test_science_ho_psf_supports_single_physical_axes() -> None:
    source = _science_samples()
    wavelength_samples = ScienceHoPsfSamples(
        zenith_angle=20.0 * u.deg,
        wavelength=source.wavelength[:2],
        x=source.x,
        y=source.y,
        psfs=source.psfs[:2],
        pixel_scale=source.pixel_scale[:2],
        tel_diameter=source.tel_diameter,
        tel_pupil=source.tel_pupil,
    )
    wavelength_artifact = build_science_ho_psf_interpolator(wavelength_samples)

    assert wavelength_artifact.coordinate_order == ("wavelength", "y", "x")
    prediction = evaluate_science_ho_psf_interpolator(
        wavelength_artifact,
        wavelength=1.5 * u.um,
        x=np.asarray([0.0]) * u.arcsec,
        y=np.asarray([0.0]) * u.arcsec,
    )
    assert prediction.pixel_scale.to_value(u.mas) == pytest.approx(6.0)

    airmass_samples = ScienceHoPsfSamples(
        zenith_angle=np.asarray([20.0, 30.0]) * u.deg,
        wavelength=1.0 * u.um,
        x=source.x,
        y=source.y,
        psfs=source.psfs[[0, 2]],
        pixel_scale=source.pixel_scale[[0, 2]],
        tel_diameter=source.tel_diameter,
        tel_pupil=source.tel_pupil,
    )
    airmass_artifact = build_science_ho_psf_interpolator(airmass_samples)

    assert airmass_artifact.coordinate_order == ("airmass", "y", "x")
    with pytest.raises(ValueError, match="zenith_angle is required"):
        evaluate_science_ho_psf_interpolator(
            airmass_artifact,
            x=np.asarray([0.0]) * u.arcsec,
            y=np.asarray([0.0]) * u.arcsec,
        )


def test_science_ho_psf_requires_only_active_physical_queries() -> None:
    interpolator = build_science_ho_psf_interpolator(_science_samples())

    with pytest.raises(ValueError, match="zenith_angle is required"):
        evaluate_science_ho_psf_interpolator(
            interpolator,
            wavelength=1.0 * u.um,
            x=np.asarray([0.0]) * u.arcsec,
            y=np.asarray([0.0]) * u.arcsec,
        )
    with pytest.raises(ValueError, match="wavelength is required"):
        evaluate_science_ho_psf_interpolator(
            interpolator,
            zenith_angle=20.0 * u.deg,
            x=np.asarray([0.0]) * u.arcsec,
            y=np.asarray([0.0]) * u.arcsec,
        )


def test_science_ho_psf_query_validation_rejects_out_of_range_before_runtime() -> None:
    interpolator = build_science_ho_psf_interpolator(_science_samples())

    with pytest.raises(ValueError, match="wavelength"):
        validate_science_ho_psf_query(
            interpolator, zenith_angle=20.0 * u.deg, wavelength=3.0 * u.um
        )
    with pytest.raises(ValueError, match="airmass"):
        validate_science_ho_psf_query(
            interpolator, zenith_angle=40.0 * u.deg, wavelength=1.5 * u.um
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
def test_science_ho_psf_fixed_coordinate_errors(
    zenith_angle: u.Quantity, wavelength: u.Quantity, message: str
) -> None:
    samples = _science_samples()
    interpolator = build_science_ho_psf_interpolator(
        ScienceHoPsfSamples(
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
        validate_science_ho_psf_query(
            interpolator, zenith_angle=zenith_angle, wavelength=wavelength
        )


def test_science_ho_psf_interpolator_rejects_malformed_grids() -> None:
    samples = _science_samples()

    with pytest.raises(ValueError, match="complete active physical-coordinate grid"):
        build_science_ho_psf_interpolator(
            ScienceHoPsfSamples(
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
        build_science_ho_psf_interpolator(
            ScienceHoPsfSamples(
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
        build_science_ho_psf_interpolator(
            ScienceHoPsfSamples(
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

    with pytest.raises(ValueError, match="meta\\['bad'\\]"):
        build_science_ho_psf_interpolator(
            ScienceHoPsfSamples(
                zenith_angle=samples.zenith_angle,
                wavelength=samples.wavelength,
                x=samples.x,
                y=samples.y,
                psfs=samples.psfs,
                pixel_scale=samples.pixel_scale,
                tel_diameter=samples.tel_diameter,
                tel_pupil=samples.tel_pupil,
                meta={"bad": np.array([1.0, 2.0]) * u.one},
            )
        )

    with pytest.raises(ValueError, match="complete rectangular"):
        build_science_ho_psf_interpolator(
            ScienceHoPsfSamples(
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

    with pytest.raises(ValueError, match="must be finite"):
        build_science_ho_psf_interpolator(
            ScienceHoPsfSamples(
                zenith_angle=samples.zenith_angle,
                wavelength=samples.wavelength,
                x=samples.x,
                y=samples.y,
                psfs=samples.psfs,
                pixel_scale=samples.pixel_scale,
                tel_diameter=samples.tel_diameter,
                tel_pupil=samples.tel_pupil,
                meta={"bad": np.nan * u.one},
            )
        )

    with pytest.raises(ValueError, match="collides with a core /meta field"):
        build_science_ho_psf_interpolator(
            ScienceHoPsfSamples(
                zenith_angle=samples.zenith_angle,
                wavelength=samples.wavelength,
                x=samples.x,
                y=samples.y,
                psfs=samples.psfs,
                pixel_scale=samples.pixel_scale,
                tel_diameter=samples.tel_diameter,
                tel_pupil=samples.tel_pupil,
                meta={"pixel_scale": 1.0 * u.mas},
            )
        )


def test_science_ho_psf_interpolator_defaults_missing_payload_meta(tmp_path) -> None:
    interpolator = build_science_ho_psf_interpolator(_science_samples())
    path = tmp_path / "missing-meta.pkl"
    save_science_ho_psf_interpolator(interpolator, path)
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    del payload["metadata"]["meta"]
    with path.open("wb") as handle:
        pickle.dump(payload, handle)

    loaded = load_science_ho_psf_interpolator(path)
    prediction = evaluate_science_ho_psf_interpolator(
        loaded,
        zenith_angle=20.0 * u.deg,
        wavelength=1.0 * u.um,
        x=_science_samples().x,
        y=_science_samples().y,
    )

    assert loaded.meta == {}
    assert prediction.meta == {}


def test_ngs_ho_metric_interpolator_round_trips_and_replays_sources(tmp_path) -> None:
    samples = _ngs_metric_samples()
    interpolator = build_ngs_ho_metric_interpolator(samples, interpolation_config=RbfInterpolationConfig(smoothing=0.0))
    path = tmp_path / "ngs.pkl"

    save_ngs_ho_metric_interpolator(interpolator, path)
    loaded = load_ngs_ho_metric_interpolator(path)
    replay = replay_ngs_ho_metric_interpolator(loaded, samples)
    prediction = evaluate_ngs_ho_metric_interpolator(
        loaded,
        zenith_angle=np.repeat(samples.zenith_angle, samples.x.size),
        x=np.tile(samples.x, samples.zenith_angle.size),
        y=np.tile(samples.y, samples.zenith_angle.size),
    )

    assert loaded.interpolation_config.smoothing == pytest.approx(0.0)
    assert type(loaded.interpolation_config).__module__ == (
        "hybrid_ao_psf.interpolation.config"
    )
    assert set(loaded.metric_names) == {"ee", "fwhm", "sr"}
    assert replay.metric_max_abs["ee"].to_value(u.one) == pytest.approx(0.0, abs=1.0e-8)
    assert replay.metric_max_abs["fwhm"].to_value(u.mas) == pytest.approx(0.0, abs=1.0e-8)
    assert replay.metric_max_abs["sr"].to_value(u.one) == pytest.approx(0.0, abs=1.0e-8)
    assert prediction.sr is not None


def test_ngs_ho_metric_interpolator_uses_regular_grid_by_default() -> None:
    interpolator = build_ngs_ho_metric_interpolator(_ngs_metric_samples())

    assert isinstance(interpolator.interpolation_config, RegularGridInterpolationConfig)
    assert interpolator.interpolation_config.method == "linear"
    assert interpolator.builder["interpolation_strategy"] == "regular_grid"
    assert set(interpolator.model["metric_grids"]) == {"ee", "fwhm", "sr"}


def test_ngs_ho_metric_regular_grid_round_trips_replays_and_interpolates(tmp_path) -> None:
    samples = _ngs_metric_samples()
    interpolator = build_ngs_ho_metric_interpolator(samples)
    path = tmp_path / "ngs-grid.pkl"

    save_ngs_ho_metric_interpolator(interpolator, path)
    loaded = load_ngs_ho_metric_interpolator(path)
    replay = replay_ngs_ho_metric_interpolator(loaded, samples)
    prediction = evaluate_ngs_ho_metric_interpolator(
        loaded,
        zenith_angle=20.0 * u.deg,
        x=np.asarray([0.0]) * u.arcsec,
        y=np.asarray([0.0]) * u.arcsec,
    )

    assert isinstance(loaded.interpolation_config, RegularGridInterpolationConfig)
    assert replay.metric_max_abs["ee"].to_value(u.one) == pytest.approx(0.0, abs=1.0e-12)
    assert replay.metric_max_abs["fwhm"].to_value(u.mas) == pytest.approx(0.0, abs=1.0e-12)
    assert replay.metric_max_abs["sr"].to_value(u.one) == pytest.approx(0.0, abs=1.0e-12)
    np.testing.assert_allclose(prediction.ee, np.asarray([0.26]) * u.one)
    np.testing.assert_allclose(prediction.fwhm, np.asarray([83.0]) * u.mas)
    np.testing.assert_allclose(prediction.sr, np.asarray([0.115]) * u.one)


def test_ngs_ho_metric_field_only_regular_grid_and_rbf_omit_zenith() -> None:
    source = _ngs_metric_samples()
    samples = NgsHoMetricSamples(
        zenith_angle=20.0 * u.deg,
        x=source.x,
        y=source.y,
        ee=source.ee[0],
        fwhm=source.fwhm[0],
        sr=source.sr[0],
    )

    regular = build_ngs_ho_metric_interpolator(samples)
    rbf = build_ngs_ho_metric_interpolator(samples, interpolation_config=RbfInterpolationConfig(smoothing=0.0))

    assert regular.coordinate_order == ("y", "x")
    assert rbf.coordinate_order == ("x", "y")
    assert "wavelength" not in regular.coordinate_order
    assert "wavelength" not in rbf.coordinate_order
    regular_prediction = evaluate_ngs_ho_metric_interpolator(
        regular,
        x=np.asarray([0.0]) * u.arcsec,
        y=np.asarray([0.0]) * u.arcsec,
    )
    rbf_prediction = evaluate_ngs_ho_metric_interpolator(
        rbf,
        x=np.asarray([-1.0]) * u.arcsec,
        y=np.asarray([-1.0]) * u.arcsec,
    )
    np.testing.assert_allclose(regular_prediction.ee, np.asarray([0.26]) * u.one)
    np.testing.assert_allclose(rbf_prediction.ee, np.asarray([0.20]) * u.one, atol=1.0e-8)
    with pytest.raises(ValueError, match="interpolator's fixed zenith_angle=20.0"):
        evaluate_ngs_ho_metric_interpolator(
            regular,
            zenith_angle=30.0 * u.deg,
            x=np.asarray([0.0]) * u.arcsec,
            y=np.asarray([0.0]) * u.arcsec,
        )


def test_ngs_ho_metric_active_airmass_requires_zenith() -> None:
    regular = build_ngs_ho_metric_interpolator(_ngs_metric_samples())
    rbf = build_ngs_ho_metric_interpolator(
        _ngs_metric_samples(),
        interpolation_config=RbfInterpolationConfig(smoothing=0.0),
    )

    assert regular.coordinate_order == ("airmass", "y", "x")
    assert rbf.coordinate_order == ("airmass", "x", "y")
    with pytest.raises(ValueError, match="zenith_angle is required"):
        evaluate_ngs_ho_metric_interpolator(
            regular,
            x=np.asarray([0.0]) * u.arcsec,
            y=np.asarray([0.0]) * u.arcsec,
        )
    with pytest.raises(ValueError, match="zenith_angle is required"):
        evaluate_ngs_ho_metric_interpolator(
            rbf,
            x=np.asarray([0.0]) * u.arcsec,
            y=np.asarray([0.0]) * u.arcsec,
        )


def test_ngs_ho_metric_interpolator_uses_explicit_rbf_default_smoothing() -> None:
    interpolator = build_ngs_ho_metric_interpolator(
        _ngs_metric_samples(),
        interpolation_config=RbfInterpolationConfig(),
    )

    assert interpolator.interpolation_config.smoothing == pytest.approx(0.05)
    assert interpolator.builder["interpolation_strategy"] == "rbf"


def test_ngs_ho_metric_interpolator_broadcasts_scalar_zenith() -> None:
    samples = _ngs_metric_samples()
    interpolator = build_ngs_ho_metric_interpolator(samples, interpolation_config=RbfInterpolationConfig(smoothing=0.0))

    prediction = evaluate_ngs_ho_metric_interpolator(
        interpolator,
        zenith_angle=20.0 * u.deg,
        x=np.asarray([-1.0, 1.0]) * u.arcsec,
        y=np.asarray([-1.0, 1.0]) * u.arcsec,
    )

    assert prediction.ee.shape == (2,)
    assert prediction.fwhm.shape == (2,)
    assert prediction.sr.shape == (2,)


def test_ngs_ho_metric_query_validation_rejects_unsupported_queries_before_evaluation() -> None:
    interpolator = build_ngs_ho_metric_interpolator(
        _ngs_metric_samples(),
        interpolation_config=RbfInterpolationConfig(smoothing=0.0),
    )

    validate_ngs_ho_metric_query(
        interpolator,
        zenith_angle=20.0 * u.deg,
        x=np.asarray([0.0]) * u.arcsec,
        y=np.asarray([0.0]) * u.arcsec,
    )
    with pytest.raises(ValueError) as above:
        validate_ngs_ho_metric_query(
            interpolator,
            zenith_angle=40.0 * u.deg,
            x=np.asarray([0.0]) * u.arcsec,
            y=np.asarray([0.0]) * u.arcsec,
        )
    assert str(above.value) == (
        "zenith_angle=40.0 deg is outside this interpolator's "
        "supported zenith_angle range [20.0, 30.0] deg."
    )
    with pytest.raises(ValueError) as below:
        validate_ngs_ho_metric_query(
            interpolator,
            zenith_angle=10.0 * u.deg,
            x=np.asarray([0.0]) * u.arcsec,
            y=np.asarray([0.0]) * u.arcsec,
        )
    assert str(below.value) == (
        "zenith_angle=10.0 deg is outside this interpolator's "
        "supported zenith_angle range [20.0, 30.0] deg."
    )
    with pytest.raises(ValueError, match="field"):
        evaluate_ngs_ho_metric_interpolator(
            interpolator,
            zenith_angle=20.0 * u.deg,
            x=np.asarray([2.0]) * u.arcsec,
            y=np.asarray([0.0]) * u.arcsec,
        )


def test_ngs_ho_metric_regular_grid_rejects_unsupported_queries_before_evaluation() -> None:
    interpolator = build_ngs_ho_metric_interpolator(_ngs_metric_samples())

    validate_ngs_ho_metric_query(
        interpolator,
        zenith_angle=20.0 * u.deg,
        x=np.asarray([0.0]) * u.arcsec,
        y=np.asarray([0.0]) * u.arcsec,
    )
    with pytest.raises(ValueError, match="zenith_angle=40.0 deg is outside"):
        validate_ngs_ho_metric_query(
            interpolator,
            zenith_angle=40.0 * u.deg,
            x=np.asarray([0.0]) * u.arcsec,
            y=np.asarray([0.0]) * u.arcsec,
        )
    with pytest.raises(ValueError, match="x"):
        evaluate_ngs_ho_metric_interpolator(
            interpolator,
            zenith_angle=20.0 * u.deg,
            x=np.asarray([2.0]) * u.arcsec,
            y=np.asarray([0.0]) * u.arcsec,
        )


def test_ngs_ho_metric_interpolator_rejects_bad_metric_values() -> None:
    samples = _ngs_metric_samples()

    with pytest.raises(ValueError, match="ee"):
        build_ngs_ho_metric_interpolator(
            NgsHoMetricSamples(
                zenith_angle=samples.zenith_angle,
                x=samples.x,
                y=samples.y,
                ee=np.full(samples.ee.shape, 1.2) * u.one,
                fwhm=samples.fwhm,
                sr=samples.sr,
            )
        )
    with pytest.raises(ValueError, match="fwhm"):
        build_ngs_ho_metric_interpolator(
            NgsHoMetricSamples(
                zenith_angle=samples.zenith_angle,
                x=samples.x,
                y=samples.y,
                ee=samples.ee,
                fwhm=np.full(samples.fwhm.shape, -1.0) * u.mas,
                sr=samples.sr,
            )
        )
    with pytest.raises(ValueError, match="sr"):
        build_ngs_ho_metric_interpolator(
            NgsHoMetricSamples(
                zenith_angle=samples.zenith_angle,
                x=samples.x,
                y=samples.y,
                ee=samples.ee,
                fwhm=samples.fwhm,
                sr=np.full(samples.sr.shape, -1.0) * u.one,
            )
        )


def test_ngs_ho_metric_samples_require_metric_quantities() -> None:
    samples = _ngs_metric_samples()

    with pytest.raises(TypeError, match="ee must be an Astropy Quantity"):
        build_ngs_ho_metric_interpolator(
            NgsHoMetricSamples(
                zenith_angle=samples.zenith_angle,
                x=samples.x,
                y=samples.y,
                ee=np.asarray(samples.ee.value),
                fwhm=samples.fwhm,
                sr=samples.sr,
            )
        )


def test_ngs_ho_metric_regular_grid_rejects_malformed_field_grids() -> None:
    samples = _ngs_metric_samples()

    with pytest.raises(ValueError, match="complete rectangular"):
        build_ngs_ho_metric_interpolator(
            NgsHoMetricSamples(
                zenith_angle=samples.zenith_angle,
                x=samples.x[:-1],
                y=samples.y[:-1],
                ee=samples.ee[:, :-1],
                fwhm=samples.fwhm[:, :-1],
                sr=samples.sr[:, :-1],
            )
        )

    with pytest.raises(ValueError, match="Duplicate NGS HO metric samples field point"):
        build_ngs_ho_metric_interpolator(
            NgsHoMetricSamples(
                zenith_angle=samples.zenith_angle,
                x=np.asarray([-1.0, -1.0, 1.0, 1.0]) * u.arcsec,
                y=np.asarray([-1.0, -1.0, 1.0, 1.0]) * u.arcsec,
                ee=samples.ee,
                fwhm=samples.fwhm,
                sr=samples.sr,
            )
        )


def test_ngs_ho_metric_interpolator_rejects_model_metric_mismatch() -> None:
    interpolator = build_ngs_ho_metric_interpolator(
        _ngs_metric_samples(),
        interpolation_config=RbfInterpolationConfig(smoothing=0.0),
    )
    model = dict(interpolator.model)
    models = dict(model["models"])
    del models["sr"]
    model["models"] = models
    bad = NgsHoMetricInterpolator(
        coordinate_order=interpolator.coordinate_order,
        zenith_angle_axis=interpolator.zenith_angle_axis,
        airmass_axis=interpolator.airmass_axis,
        x=interpolator.x,
        y=interpolator.y,
        metric_names=interpolator.metric_names,
        interpolation_config=interpolator.interpolation_config,
        model=model,
        provenance=interpolator.provenance,
        builder=interpolator.builder,
    )

    with pytest.raises(ValueError, match="model does not match metric_names"):
        validate_ngs_ho_metric_interpolator(bad)


def test_ngs_ho_psfs_feed_metric_samples_then_metric_interpolator() -> None:
    psf_samples = _ngs_psf_samples()
    metric_samples = build_ngs_ho_metric_samples_from_psfs(psf_samples)
    direct = build_ngs_ho_metric_interpolator(metric_samples, interpolation_config=RbfInterpolationConfig(smoothing=0.0))
    from_psfs = build_ngs_ho_metric_interpolator_from_psfs(
        psf_samples,
        interpolation_config=RbfInterpolationConfig(smoothing=0.0),
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


def test_ngs_ho_psf_samples_require_full_psf_metadata() -> None:
    samples = _ngs_psf_samples()

    with pytest.raises(ValueError, match="tel_pupil"):
        build_ngs_ho_metric_samples_from_psfs(
            NgsHoPsfSamples(
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


def _science_samples() -> ScienceHoPsfSamples:
    zenith = np.asarray([20.0, 20.0, 30.0, 30.0])
    wavelength = np.asarray([1.0, 2.0, 1.0, 2.0])
    x = np.asarray([-1.0, 1.0, -1.0, 1.0])
    y = np.asarray([-1.0, -1.0, 1.0, 1.0])
    psfs = np.empty((4, 4, 9, 9), dtype=np.float32)
    for plane in range(4):
        for point, (px, py) in enumerate(zip(x, y, strict=True)):
            psfs[plane, point] = (
                (2.0 + 0.5 * plane + 0.25 * point)
                * _gaussian_psf(9, sigma=1.0 + 0.05 * plane, dx=0.1 * px, dy=0.1 * py)
            )
    return ScienceHoPsfSamples(
        zenith_angle=zenith * u.deg,
        wavelength=wavelength * u.um,
        x=x * u.arcsec,
        y=y * u.arcsec,
        psfs=psfs,
        pixel_scale=np.asarray([4.0, 8.0, 4.5, 9.0]) * u.mas,
        tel_diameter=30.0 * u.m,
        tel_pupil=np.ones((4, 4), dtype=np.float32) * u.dimensionless_unscaled,
        meta={
            "norm_correction": 0.75 * u.dimensionless_unscaled,
            "plane_correction": np.asarray([1.0, 2.0, 3.0, 4.0])
            * u.dimensionless_unscaled,
        },
        provenance=("synthetic",),
    )


def _ngs_metric_samples() -> NgsHoMetricSamples:
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
    return NgsHoMetricSamples(
        zenith_angle=zenith * u.deg,
        x=x * u.arcsec,
        y=y * u.arcsec,
        ee=ee * u.dimensionless_unscaled,
        fwhm=fwhm * u.mas,
        sr=sr * u.dimensionless_unscaled,
        provenance=("synthetic",),
    )


def _ngs_psf_samples() -> NgsHoPsfSamples:
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
    return NgsHoPsfSamples(
        zenith_angle=zenith * u.deg,
        x=x * u.arcsec,
        y=y * u.arcsec,
        psfs=psfs,
        wavelength=np.asarray([1.65, 1.65]) * u.um,
        pixel_scale=np.asarray([5.0, 5.0]) * u.mas,
        tel_diameter=30.0 * u.m,
        tel_pupil=np.ones((8, 8), dtype=np.float32) * u.dimensionless_unscaled,
        provenance=("synthetic",),
    )


def _gaussian_psf(size: int, *, sigma: float, dx: float = 0.0, dy: float = 0.0) -> np.ndarray:
    coords = np.arange(size, dtype=float) - (size - 1) / 2.0
    xx, yy = np.meshgrid(coords - dx, coords - dy)
    psf = np.exp(-0.5 * (xx**2 + yy**2) / sigma**2)
    return (psf / psf.sum()).astype(np.float32)
