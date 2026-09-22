from pathlib import Path

import numpy as np
import pytest
from astropy import units as u

from hybrid_ao_psf import DiagnosticsLevel, NonPsdPolicy, load_configuration


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
science_psf_artifact = science.pkl
ngs_metric_artifact = ngs.pkl
output = results/result.h5
ngs_flux_photon_per_s = {flux}
non_psd_policy = clip
diagnostics_level = debug
{hybrid_extra}
""",
        encoding="utf-8",
    )


def test_load_configuration_resolves_paths_and_cartesian_coordinates(tmp_path: Path) -> None:
    path = tmp_path / "case" / "simulation.ini"
    path.parent.mkdir()
    _write_config(path)

    configuration = load_configuration(path)

    assert configuration.source == path.resolve()
    assert configuration.science_psf_artifact == (path.parent / "science.pkl").resolve()
    assert configuration.ngs_metric_artifact == (path.parent / "ngs.pkl").resolve()
    assert configuration.output == (path.parent / "results/result.h5").resolve()
    np.testing.assert_allclose(
        configuration.request.science_x.to_value(u.arcsec), [0.0, 0.0], atol=1e-14
    )
    np.testing.assert_allclose(
        configuration.request.science_y.to_value(u.arcsec), [0.0, 2.0], atol=1e-14
    )
    np.testing.assert_allclose(
        configuration.request.ngs_x.to_value(u.arcsec), [10.0, -20.0], atol=1e-14
    )
    assert configuration.request.wavelength.to_value(u.um) == pytest.approx(1.65)
    assert configuration.request.non_psd_policy is NonPsdPolicy.CLIP
    assert configuration.request.diagnostics_level is DiagnosticsLevel.DEBUG
    assert "[hybrid]" not in configuration.request.mastsel_ini.lower()
    assert "[telescope]" in configuration.request.mastsel_ini


def test_load_configuration_rejects_ngs_flux_shape(tmp_path: Path) -> None:
    path = tmp_path / "simulation.ini"
    _write_config(path, flux="[1000]")

    with pytest.raises(ValueError, match="must match sources_LO"):
        load_configuration(path)


def test_load_configuration_rejects_independent_pixel_scale(tmp_path: Path) -> None:
    path = tmp_path / "simulation.ini"
    _write_config(path, hybrid_extra="pixel_scale_mas = 4.0")

    with pytest.raises(ValueError, match="science PSF artifact owns the pixel scale"):
        load_configuration(path)
