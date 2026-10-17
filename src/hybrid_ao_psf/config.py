"""Standalone INI loading for explicit peer low-order models."""

from __future__ import annotations

import ast
from configparser import ConfigParser, SectionProxy
from dataclasses import dataclass
from io import StringIO
from pathlib import Path

import numpy as np
from astropy import units as u

from ._calibration_config import load_e2e_calibration_configuration
from .contracts import DiagnosticsLevel, HybridRequest, LowOrderModel, NonPsdPolicy

HYBRID_SECTION = "hybrid"


@dataclass(frozen=True)
class HybridConfiguration:
    """Resolved standalone configuration and its external file paths.

    Attributes:
        request: Validated engine request derived from the combined INI file.
        science_psf_artifact: Science high-order PSF provider artifact.
        ngs_metric_artifact: External MASTSEL metric provider path; None for E2E,
            whose metrics belong to the loaded calibrated model.
        output: Configured standalone result path.
        source: Absolute source INI path.
        e2e_configuration: Shared calibration/model INI for E2E; None for MASTSEL.
            Loading configuration does not acquire native data or load fitted models.
    """

    request: HybridRequest
    science_psf_artifact: Path
    ngs_metric_artifact: Path | None
    output: Path
    source: Path
    e2e_configuration: Path | None = None


def load_configuration(path: str | Path, *, dtype=None) -> HybridConfiguration:
    """Resolve one INI into a model-specific request and explicit file paths.

    MASTSEL is the default and retains its combined INI/polar source format.
    E2E uses Cartesian [hybrid] request fields plus the shared [e2e] model and
    provider paths. Relative paths resolve from this INI. Precision resolves
    from model-specific INI defaults then explicit dtype; no second policy is
    inferred from file presence. Missing/unsupported request inputs raise.
    This operation does not load fitted models or start calibration.
    """
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Hybrid AO PSF configuration not found: {source}")
    parser = ConfigParser(interpolation=None)
    parser.optionxform = str
    with source.open(encoding="utf-8") as stream:
        parser.read_file(stream)
    if not parser.has_section(HYBRID_SECTION):
        raise ValueError("Configuration requires [hybrid].")
    model = LowOrderModel(
        parser[HYBRID_SECTION].get("lo_model", LowOrderModel.MASTSEL.value)
    )
    if model is LowOrderModel.E2E_CALIBRATED:
        return _load_calibrated_configuration(parser, source, dtype)
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


def _load_calibrated_configuration(parser, source, dtype):
    """Parse Cartesian prediction inputs and the shared calibration INI paths."""
    hybrid = parser[HYBRID_SECTION]
    allowed = {
        "lo_model",
        "output",
        "science_x_arcsec",
        "science_y_arcsec",
        "ngs_x_arcsec",
        "ngs_y_arcsec",
        "ngs_magnitude",
        "wavelength_um",
        "zenith_angle_deg",
        "diagnostics_level",
        "non_psd_policy",
    }
    unknown = set(hybrid) - allowed
    if unknown:
        raise ValueError(
            f"unsupported E2E [hybrid] keys: {sorted(unknown)}; precision belongs in [e2e]"
        )
    calibrated = load_e2e_calibration_configuration(source)
    request = HybridRequest(
        lo_model=LowOrderModel.E2E_CALIBRATED,
        dtype=calibrated.dtype if dtype is None else dtype,
        science_x=_array(hybrid, "science_x_arcsec", label="hybrid.science_x_arcsec")
        * u.arcsec,
        science_y=_array(hybrid, "science_y_arcsec", label="hybrid.science_y_arcsec")
        * u.arcsec,
        ngs_x=_array(hybrid, "ngs_x_arcsec", label="hybrid.ngs_x_arcsec") * u.arcsec,
        ngs_y=_array(hybrid, "ngs_y_arcsec", label="hybrid.ngs_y_arcsec") * u.arcsec,
        ngs_magnitude=_array(hybrid, "ngs_magnitude", label="hybrid.ngs_magnitude")
        * u.mag,
        wavelength=_scalar(hybrid, "wavelength_um", label="hybrid.wavelength_um")
        * u.um,
        zenith_angle=_scalar(
            hybrid, "zenith_angle_deg", label="hybrid.zenith_angle_deg"
        )
        * u.deg,
        diagnostics_level=hybrid.get("diagnostics_level", DiagnosticsLevel.NONE.value),
        non_psd_policy=hybrid.get("non_psd_policy", NonPsdPolicy.ERROR.value),
    )
    if not calibrated.science_psf.is_file():
        raise FileNotFoundError(
            f"science PSF artifact not found: {calibrated.science_psf}"
        )
    return HybridConfiguration(
        request=request,
        science_psf_artifact=calibrated.science_psf,
        ngs_metric_artifact=None,
        output=_resolved_path(hybrid, "output", source=source, must_exist=False),
        source=source,
        e2e_configuration=source,
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
