"""Combined Hybrid-and-MASTSEL INI configuration loading."""

from __future__ import annotations

import ast
from configparser import ConfigParser, SectionProxy
from dataclasses import dataclass
from io import StringIO
from pathlib import Path

import numpy as np
from astropy import units as u

from .contracts import DiagnosticsLevel, HybridRequest, NonPsdPolicy

HYBRID_SECTION = "hybrid"


@dataclass(frozen=True)
class HybridConfiguration:
    """Resolved standalone configuration and its external file paths.

    Attributes:
        request: Validated engine request derived from the combined INI file.
        science_psf_artifact: Science high-order PSF provider artifact.
        ngs_metric_artifact: NGS high-order metric provider artifact.
        output: Configured standalone result path.
        source: Absolute path of the source INI file.
    """

    request: HybridRequest
    science_psf_artifact: Path
    ngs_metric_artifact: Path
    output: Path
    source: Path


def load_configuration(path: str | Path, *, dtype=None) -> HybridConfiguration:
    """Load one combined Hybrid-and-MASTSEL INI file.

    Relative artifact and output paths are resolved from the directory that
    contains the INI file. MASTSEL's polar source coordinates are normalized
    into the Cartesian request contract.
    """
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Hybrid AO PSF configuration not found: {source}")
    parser = ConfigParser(interpolation=None)
    parser.optionxform = str
    with source.open(encoding="utf-8") as stream:
        parser.read_file(stream)
    for section in (
        HYBRID_SECTION,
        "telescope",
        "sources_science",
        "sources_LO",
        "RTC",
    ):
        if not parser.has_section(section):
            raise ValueError(f"Configuration requires [{section}].")

    hybrid = parser[HYBRID_SECTION]
    if "pixel_scale_mas" in hybrid:
        raise ValueError(
            "hybrid.pixel_scale_mas is not supported; the science PSF artifact "
            "owns the pixel scale."
        )
    science_r = _array(
        parser["sources_science"], "Zenith", label="sources_science.Zenith"
    )
    science_theta = _array(
        parser["sources_science"], "Azimuth", label="sources_science.Azimuth"
    )
    science_x, science_y = _polar_to_cartesian(science_r, science_theta)
    ngs_r = _array(parser["sources_LO"], "Zenith", label="sources_LO.Zenith")
    ngs_theta = _array(parser["sources_LO"], "Azimuth", label="sources_LO.Azimuth")
    ngs_x, ngs_y = _polar_to_cartesian(ngs_r, ngs_theta)

    wavelength_values = _array(
        parser["sources_science"],
        "Wavelength",
        label="sources_science.Wavelength",
    )
    if wavelength_values.size != 1:
        raise ValueError("sources_science.Wavelength must contain exactly one value.")
    ngs_flux = _array(
        hybrid, "ngs_flux_photon_per_s", label="hybrid.ngs_flux_photon_per_s"
    )
    if ngs_flux.shape != ngs_r.shape:
        raise ValueError(
            "hybrid.ngs_flux_photon_per_s must match sources_LO coordinates."
        )
    request = HybridRequest(
        dtype=hybrid.get("precision", "float32") if dtype is None else dtype,
        science_x=science_x * u.arcsec,
        science_y=science_y * u.arcsec,
        ngs_x=ngs_x * u.arcsec,
        ngs_y=ngs_y * u.arcsec,
        ngs_flux=ngs_flux * (u.photon / u.s),
        wavelength=float(wavelength_values[0]) * u.m,
        zenith_angle=_scalar(
            parser["telescope"], "ZenithAngle", label="telescope.ZenithAngle"
        )
        * u.deg,
        mastsel_ini=_mastsel_ini(parser),
        non_psd_policy=hybrid.get("non_psd_policy", NonPsdPolicy.ERROR.value),
        diagnostics_level=hybrid.get("diagnostics_level", DiagnosticsLevel.NONE.value),
    )
    return HybridConfiguration(
        request=request,
        science_psf_artifact=_resolved_path(
            hybrid,
            "science_psf_artifact",
            source=source,
        ),
        ngs_metric_artifact=_resolved_path(
            hybrid,
            "ngs_metric_artifact",
            source=source,
        ),
        output=_resolved_path(hybrid, "output", source=source, must_exist=False),
        source=source,
    )


def _resolved_path(
    section: SectionProxy,
    option: str,
    *,
    source: Path,
    must_exist: bool = True,
) -> Path:
    raw = section.get(option, "").strip()
    if not raw:
        raise ValueError(f"hybrid.{option} must not be empty.")
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = source.parent / path
    path = path.resolve()
    if must_exist and not path.is_file():
        raise FileNotFoundError(
            f"Hybrid AO PSF artifact not found for hybrid.{option}: {path}"
        )
    return path


def _array(section: SectionProxy, option: str, *, label: str) -> np.ndarray:
    if option not in section:
        raise ValueError(f"Configuration requires {label}.")
    raw = section.get(option, "")
    try:
        parsed = ast.literal_eval(raw)
    except (SyntaxError, ValueError) as exc:
        raise ValueError(f"{label} must be a numeric scalar or list.") from exc
    values = np.asarray(parsed, dtype=float)
    if values.ndim == 0:
        values = values.reshape(1)
    if values.ndim != 1 or values.size == 0:
        raise ValueError(f"{label} must be a non-empty one-dimensional list.")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{label} must contain only finite values.")
    return values


def _scalar(section: SectionProxy, option: str, *, label: str) -> float:
    if option not in section:
        raise ValueError(f"Configuration requires {label}.")
    try:
        value = float(section[option])
    except ValueError as exc:
        raise ValueError(f"{label} must be a finite scalar.") from exc
    if not np.isfinite(value):
        raise ValueError(f"{label} must be a finite scalar.")
    return value


def _polar_to_cartesian(
    r: np.ndarray, theta: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    if r.shape != theta.shape:
        raise ValueError(
            "Polar coordinate radius and azimuth vectors must have identical shapes."
        )
    radians = np.deg2rad(theta)
    return r * np.cos(radians), r * np.sin(radians)


def _mastsel_ini(parser: ConfigParser) -> str:
    mastsel = ConfigParser(interpolation=None)
    mastsel.optionxform = str
    for section in parser.sections():
        if section.casefold() == HYBRID_SECTION:
            continue
        mastsel.add_section(section)
        for option, value in parser.items(section, raw=True):
            mastsel.set(section, option, value)
    stream = StringIO()
    mastsel.write(stream)
    return stream.getvalue()


__all__ = ["HYBRID_SECTION", "HybridConfiguration", "load_configuration"]
