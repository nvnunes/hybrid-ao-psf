from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from astropy import units as u

from hybrid_ao_psf import (
    DiagnosticsLevel,
    MastselModel,
    NonPsdPolicy,
    _mastsel,
    engine,
    load_configuration,
)


def _write_config(
    path: Path,
    *,
    flux: str = "[1000, 2000]",
    hybrid_extra: str = "",
) -> None:
    (path.parent / "science.pkl").touch()
    (path.parent / "ngs.pkl").touch()
    path.write_text(
        f"""[telescope]
TelescopeDiameter = 8.0
ZenithAngle = 30.0

[sources_science]
Wavelength = [1.65e-6]
Zenith = [0.0, 2.0]
Azimuth = [0.0, 90.0]

[sources_LO]
Wavelength = [710e-9]
Zenith = [10.0, 20.0]
Azimuth = [0.0, 180.0]

[RTC]
SensorFrameRate_LO = 500.0

[hybrid]
science_psf_file = science.pkl
ngs_metric_file = ngs.pkl
output = results/result.h5
ngs_flux_photon_per_s = {flux}
non_psd_policy = clip
diagnostics_level = debug
{hybrid_extra}
""",
        encoding="utf-8",
    )


def test_load_configuration_resolves_paths_and_cartesian_coordinates(
    tmp_path: Path,
) -> None:
    path = tmp_path / "case" / "simulation.ini"
    path.parent.mkdir()
    _write_config(path)

    configuration = load_configuration(path)

    assert configuration.source == path.resolve()
    assert configuration.science_psf_file == (path.parent / "science.pkl").resolve()
    assert configuration.ngs_metric_file == (path.parent / "ngs.pkl").resolve()
    assert configuration.output == (path.parent / "results/result.h5").resolve()
    np.testing.assert_allclose(
        configuration.request.science_positions.to_value(u.arcsec),
        [[0.0, 0.0], [0.0, 2.0]],
        atol=1e-14,
    )
    np.testing.assert_allclose(
        configuration.request.ngs_positions.to_value(u.arcsec),
        [[10.0, 0.0], [-20.0, 0.0]],
        atol=1e-14,
    )
    assert configuration.request.wavelength.to_value(u.um) == pytest.approx(1.65)
    assert configuration.non_psd_policy is NonPsdPolicy.CLIP
    assert configuration.request.diagnostics_level is DiagnosticsLevel.DEBUG
    assert "[hybrid]" not in configuration.mastsel_ini.lower()
    assert "[telescope]" in configuration.mastsel_ini
    assert "SensorFrameRate_LO = 500.0" in configuration.mastsel_ini
    assert not hasattr(configuration.request, "ngs_frame_rate")


def test_load_configuration_rejects_ngs_flux_shape(tmp_path: Path) -> None:
    path = tmp_path / "simulation.ini"
    _write_config(path, flux="[1000]")

    with pytest.raises(ValueError, match="must match sources_LO"):
        load_configuration(path)


def test_load_configuration_rejects_independent_pixel_scale(tmp_path: Path) -> None:
    path = tmp_path / "simulation.ini"
    _write_config(path, hybrid_extra="pixel_scale_mas = 4.0")

    with pytest.raises(ValueError, match="science PSF file owns the pixel scale"):
        load_configuration(path)


def test_previous_interpolator_path_keys_still_load(tmp_path: Path) -> None:
    path = tmp_path / "simulation.ini"
    _write_config(path)
    text = path.read_text().replace("science_psf_file", "science_psf_artifact")
    path.write_text(text.replace("ngs_metric_file", "ngs_metric_artifact"))
    configuration = load_configuration(path)
    assert configuration.science_psf_file == (tmp_path / "science.pkl").resolve()
    assert configuration.ngs_metric_file == (tmp_path / "ngs.pkl").resolve()


def test_conflicting_previous_interpolator_path_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "simulation.ini"
    _write_config(path, hybrid_extra="science_psf_artifact = other.pkl")
    with pytest.raises(ValueError, match="Conflicting hybrid.science_psf_file"):
        load_configuration(path)


def test_mastsel_model_parses_once_and_isolates_each_run(tmp_path, monkeypatch):
    path = tmp_path / "simulation.ini"
    _write_config(path)
    configuration = load_configuration(path)
    parse = _mastsel.parse_ini
    parsed = []

    def record_parse(text):
        parsed.append(text)
        return parse(text)

    monkeypatch.setattr(_mastsel, "parse_ini", record_parse)
    model = MastselModel(
        mastsel_ini=configuration.mastsel_ini,
        ngs_metrics=SimpleNamespace(evaluate=lambda **kwargs: None),
    )
    first, _, _, _ = engine._resolve_effective_mastsel(configuration.request, model)
    first["RTC"]["SensorFrameRate_LO"] = "100"
    second, rates, _, _ = engine._resolve_effective_mastsel(
        configuration.request, model
    )
    assert parsed == [model.mastsel_ini]
    assert second["RTC"]["SensorFrameRate_LO"] == "500.0"
    np.testing.assert_array_equal(rates.to_value(u.Hz), [500, 500])

    override = configuration.mastsel_ini.replace(
        "SensorFrameRate_LO = 500.0", "SensorFrameRate_LO = 200"
    )
    edited = replace(model, mastsel_ini=override)
    _, rates, _, _ = engine._resolve_effective_mastsel(configuration.request, edited)
    np.testing.assert_array_equal(rates.to_value(u.Hz), [200, 200])
    assert parsed == [model.mastsel_ini, edited.mastsel_ini]
