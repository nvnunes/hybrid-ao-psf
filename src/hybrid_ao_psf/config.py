"""Standalone INI loading for explicit peer low-order models."""

from __future__ import annotations

from configparser import ConfigParser
from dataclasses import dataclass
from io import StringIO
from pathlib import Path

import numpy as np
from ao_core.models.config import (
    parse_ini,
    polar_to_cartesian,
    read_ini,
    read_ini_array,
    read_ini_scalar,
    resolve_ini_path,
)
from astropy import units as u
from numpy.typing import DTypeLike

from ._calibration_config import _read_e2e_calibration_configuration
from .contracts import DiagnosticsLevel, HybridRequest, LowOrderModel, NonPsdPolicy

HYBRID_SECTION = "hybrid"

type _IniSections = tuple[tuple[str, tuple[tuple[str, str], ...]], ...]


@dataclass(frozen=True)
class HybridConfiguration:
    """Resolved standalone configuration and its external file paths.

    Attributes:
        request: Validated observation derived from the combined INI file.
        lo_model: Model selected by the INI for standalone execution.
        mastsel_ini: MASTSEL configuration text for MASTSEL runs.
        non_psd_policy: Covariance conditioning policy for MASTSEL.
        science_psf_file: Science high-order PSF provider file.
        ngs_metric_file: External MASTSEL metric provider path; None for E2E,
            whose metrics belong to the loaded calibrated model.
        output: Configured standalone result path.
        source: Absolute source INI path.
        e2e_configuration: Shared calibration/model INI for E2E; None for MASTSEL.
            Loading configuration does not acquire native data or load fitted models.
    """

    request: HybridRequest
    science_psf_file: Path
    ngs_metric_file: Path | None
    output: Path
    source: Path
    lo_model: LowOrderModel
    e2e_configuration: Path | None = None
    mastsel_ini: str | None = None
    non_psd_policy: NonPsdPolicy = NonPsdPolicy.ERROR


def load_configuration(
    path: str | Path, *, dtype: DTypeLike | None = None
) -> HybridConfiguration:
    """Read a Hybrid INI and resolve its model-specific inputs.

    Paths in the INI are relative to the INI's directory. MASTSEL is the
    default and reads the combined INI with polar source coordinates.
    E2E-Calibrated reads Cartesian coordinates and the shared `[e2e]` section.
    The returned configuration carries file paths for later provider loading.

    Args:
        path: Path to the Hybrid configuration.
        dtype: Optional float32 or float64 precision override. By default,
            use the configured model precision.

    Returns:
        A validated request, provider paths and the output destination.

    Raises:
        FileNotFoundError: If the INI or required path is missing.
        ValueError: If the model, required fields or precision are invalid.
    """
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Hybrid AO PSF configuration not found: {source}")
    source, parser = read_ini(source)
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
    for option in ("science_psf_file", "ngs_metric_file"):
        previous = option.removesuffix("_file") + "_artifact"
        if previous in hybrid:
            if option in hybrid and hybrid[option] != hybrid[previous]:
                raise ValueError(f"Conflicting hybrid.{option} paths.")
            hybrid[option] = hybrid.pop(previous)
    if "pixel_scale_mas" in hybrid:
        raise ValueError(
            "hybrid.pixel_scale_mas is not supported; the science PSF file "
            "owns the pixel scale."
        )
    science_r = read_ini_array(
        parser["sources_science"], "Zenith", label="sources_science.Zenith"
    )
    science_theta = read_ini_array(
        parser["sources_science"], "Azimuth", label="sources_science.Azimuth"
    )
    science_x, science_y = polar_to_cartesian(science_r, science_theta)
    ngs_r = read_ini_array(parser["sources_LO"], "Zenith", label="sources_LO.Zenith")
    ngs_theta = read_ini_array(
        parser["sources_LO"], "Azimuth", label="sources_LO.Azimuth"
    )
    ngs_x, ngs_y = polar_to_cartesian(ngs_r, ngs_theta)

    wavelength_values = read_ini_array(
        parser["sources_science"],
        "Wavelength",
        label="sources_science.Wavelength",
    )
    if wavelength_values.size != 1:
        raise ValueError("sources_science.Wavelength must contain exactly one value.")
    ngs_flux = read_ini_array(
        hybrid, "ngs_flux_photon_per_s", label="hybrid.ngs_flux_photon_per_s"
    )
    if ngs_flux.shape != ngs_r.shape:
        raise ValueError(
            "hybrid.ngs_flux_photon_per_s must match sources_LO coordinates."
        )
    mastsel_sections = tuple(
        (section, tuple(parser.items(section, raw=True)))
        for section in parser.sections()
        if section.casefold() != HYBRID_SECTION
    )
    stream = StringIO()
    _mastsel_parser(mastsel_sections).write(stream)
    request = HybridRequest(
        dtype=hybrid.get("precision", "float32") if dtype is None else dtype,
        science_positions=np.column_stack((science_x, science_y)) * u.arcsec,
        ngs_positions=np.column_stack((ngs_x, ngs_y)) * u.arcsec,
        ngs_flux=ngs_flux * (u.photon / u.s),
        wavelength=float(wavelength_values[0]) * u.m,
        zenith_angle=read_ini_scalar(
            parser["telescope"], "ZenithAngle", label="telescope.ZenithAngle"
        )
        * u.deg,
        diagnostics_level=hybrid.get("diagnostics_level", DiagnosticsLevel.NONE.value),
    )
    return HybridConfiguration(
        request=request,
        lo_model=LowOrderModel.MASTSEL,
        mastsel_ini=stream.getvalue(),
        non_psd_policy=NonPsdPolicy(
            hybrid.get("non_psd_policy", NonPsdPolicy.ERROR.value)
        ),
        science_psf_file=_resolved_path(
            hybrid,
            "science_psf_file",
            source=source,
        ),
        ngs_metric_file=_resolved_path(
            hybrid,
            "ngs_metric_file",
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
    calibrated = _read_e2e_calibration_configuration(parser, source)
    policy = NonPsdPolicy(hybrid.get("non_psd_policy", NonPsdPolicy.ERROR.value))
    if policy is not NonPsdPolicy.ERROR:
        raise ValueError(
            "E2E-calibrated configuration requires non_psd_policy = error."
        )
    request = HybridRequest(
        dtype=calibrated.dtype if dtype is None else dtype,
        science_positions=_read_cartesian_positions(hybrid, "science"),
        ngs_positions=_read_cartesian_positions(hybrid, "ngs"),
        ngs_magnitude=read_ini_array(
            hybrid, "ngs_magnitude", label="hybrid.ngs_magnitude"
        )
        * u.mag,
        wavelength=read_ini_scalar(
            hybrid, "wavelength_um", label="hybrid.wavelength_um"
        )
        * u.um,
        zenith_angle=read_ini_scalar(
            hybrid, "zenith_angle_deg", label="hybrid.zenith_angle_deg"
        )
        * u.deg,
        diagnostics_level=hybrid.get("diagnostics_level", DiagnosticsLevel.NONE.value),
    )
    if not calibrated.science_psf.is_file():
        raise FileNotFoundError(f"science PSF file not found: {calibrated.science_psf}")
    return HybridConfiguration(
        request=request,
        lo_model=LowOrderModel.E2E_CALIBRATED,
        science_psf_file=calibrated.science_psf,
        ngs_metric_file=None,
        output=_resolved_path(hybrid, "output", source=source, must_exist=False),
        source=source,
        e2e_configuration=source,
    )


def _read_cartesian_positions(section, field: str) -> u.Quantity:
    """Read the INI's x and y vectors as a single array of angular pairs."""
    x_key, y_key = f"{field}_x_arcsec", f"{field}_y_arcsec"
    x = read_ini_array(section, x_key, label=f"hybrid.{x_key}")
    y = read_ini_array(section, y_key, label=f"hybrid.{y_key}")
    if x.shape != y.shape:
        raise ValueError(f"hybrid.{x_key} and hybrid.{y_key} must have the same length.")
    return np.column_stack((x, y)) * u.arcsec


def _resolved_path(section, option, *, source, must_exist=True) -> Path:
    return resolve_ini_path(
        section.get(option, ""),
        source=source,
        label=f"hybrid.{option}",
        must_exist=must_exist,
    )


def _mastsel_parser(sections: _IniSections) -> ConfigParser:
    mastsel = parse_ini("")
    mastsel.read_dict({section: dict(options) for section, options in sections})
    return mastsel


__all__ = ["HYBRID_SECTION", "HybridConfiguration", "load_configuration"]
