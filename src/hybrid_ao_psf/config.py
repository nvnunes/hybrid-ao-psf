"""Standalone INI loading for explicit peer low-order models."""

from __future__ import annotations

from configparser import ConfigParser, SectionProxy
from dataclasses import dataclass
from functools import partial
from io import StringIO
from pathlib import Path

import numpy as np
from ao_core.models.config import (
    Configuration,
    IniSource,
    load_ao_configuration,
    parse_ini,
    read_ini_array,
    read_ini_scalar,
    resolve_ini_path,
)
from ao_core.photometry import WFSPhotometryConfig, photons_per_frame_to_magnitudes
from astropy import units as u
from numpy.typing import DTypeLike

from ._calibration_config import _read_e2e_calibration_configuration
from .contracts import (
    DiagnosticsLevel,
    HybridRequest,
    LowOrderModel,
    NonPsdPolicy,
    _positive_scalar,
)

_HYBRID_SECTION = "hybrid"

_ZEROPOINT_UNIT = u.photon / (u.m**2 * u.s)
_LEGACY_FLUX_ZEROPOINT = 1e10 * _ZEROPOINT_UNIT

type _IniSections = tuple[tuple[str, tuple[tuple[str, str], ...]], ...]


@dataclass(frozen=True)
class HybridConfiguration(Configuration):
    """Shared AO inputs plus the resolved standalone Hybrid settings.

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
        base_config: Original INI text, before resolving run overrides.
        ngs_magnitude_zeropoint: MASTSEL zero-magnitude flux before WFS
            area scaling, in photon per square metre per second.
        ngs_mag_standard: Magnitude convention, if supplied in the INI.
        e2e_configuration: Shared calibration/model INI for E2E; None for MASTSEL.
            Loading configuration does not acquire native data or load fitted models.
    """

    request: HybridRequest
    science_psf_file: Path
    ngs_metric_file: Path | None
    output: Path
    source: Path
    lo_model: LowOrderModel
    base_config: str
    ngs_magnitude_zeropoint: u.Quantity | None = None
    ngs_mag_standard: str | None = None
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
    return load_ao_configuration(
        source, load_pupil=False, decode=partial(_decode_configuration, dtype=dtype)
    )


def _decode_configuration(
    ini_source: IniSource, ao_config: Configuration, *, dtype: DTypeLike | None
) -> HybridConfiguration:
    ini_path, base_ini_text, parser = ini_source.path, ini_source.text, ini_source.parser
    assert ini_path is not None and base_ini_text is not None
    if not parser.has_section(_HYBRID_SECTION):
        raise ValueError("Configuration requires [hybrid].")
    low_order_model = LowOrderModel(
        parser[_HYBRID_SECTION].get("lo_model", LowOrderModel.MASTSEL.value)
    )
    if low_order_model is LowOrderModel.E2E_CALIBRATED:
        return _decode_calibrated_configuration(ini_source, ao_config, dtype)
    for section in (
        _HYBRID_SECTION,
        "telescope",
        "sources_science",
        "sources_LO",
        "RTC",
    ):
        if not parser.has_section(section):
            raise ValueError(f"Configuration requires [{section}].")

    hybrid_section = parser[_HYBRID_SECTION]
    for option in ("science_psf_file", "ngs_metric_file"):
        previous = option.removesuffix("_file") + "_artifact"
        if previous in hybrid_section:
            if (
                option in hybrid_section
                and hybrid_section[option] != hybrid_section[previous]
            ):
                raise ValueError(f"Conflicting hybrid.{option} paths.")
            hybrid_section[option] = hybrid_section.pop(previous)
    if "pixel_scale_mas" in hybrid_section:
        raise ValueError(
            "hybrid.pixel_scale_mas is not supported; the science PSF file "
            "owns the pixel scale."
        )
    science_sources = ao_config.science_sources
    low_order_sources = ao_config.low_order_sources
    telescope_config = ao_config.telescope
    if science_sources is None or science_sources.directions is None:
        raise ValueError("Configuration requires sources_science.Zenith and Azimuth.")
    if low_order_sources is None or low_order_sources.directions is None:
        raise ValueError("Configuration requires sources_LO.Zenith and Azimuth.")
    science_wavelengths = science_sources.wavelengths
    if science_wavelengths is None:
        raise ValueError("Configuration requires sources_science.Wavelength.")
    if science_wavelengths.size != 1:
        raise ValueError("sources_science.Wavelength must contain exactly one value.")
    if telescope_config is None or telescope_config.zenith_angle is None:
        raise ValueError("Configuration requires telescope.ZenithAngle.")
    magnitudes, zeropoint, mag_standard = _read_mastsel_magnitudes(
        hybrid_section, ao_config, (len(low_order_sources.directions),)
    )
    mastsel_sections = tuple(
        (section, tuple(parser.items(section, raw=True)))
        for section in parser.sections()
        if section.casefold() != _HYBRID_SECTION
    )
    stream = StringIO()
    _mastsel_parser(mastsel_sections).write(stream)
    request = HybridRequest(
        dtype=hybrid_section.get("precision", "float32") if dtype is None else dtype,
        science_positions=science_sources.directions,
        ngs_positions=low_order_sources.directions,
        ngs_magnitude=magnitudes,
        wavelength=science_wavelengths[0],
        zenith_angle=telescope_config.zenith_angle,
        diagnostics_level=hybrid_section.get(
            "diagnostics_level", DiagnosticsLevel.NONE.value
        ),
    )
    return HybridConfiguration.from_config(
        ao_config,
        request=request,
        lo_model=LowOrderModel.MASTSEL,
        mastsel_ini=stream.getvalue(),
        non_psd_policy=NonPsdPolicy(
            hybrid_section.get("non_psd_policy", NonPsdPolicy.ERROR.value)
        ),
        science_psf_file=_resolved_path(
            hybrid_section,
            "science_psf_file",
            source=ini_path,
        ),
        ngs_metric_file=_resolved_path(
            hybrid_section,
            "ngs_metric_file",
            source=ini_path,
        ),
        output=_resolved_path(hybrid_section, "output", source=ini_path, must_exist=False),
        source=ini_path,
        base_config=base_ini_text,
        ngs_magnitude_zeropoint=zeropoint,
        ngs_mag_standard=mag_standard,
    )


def _read_mastsel_magnitudes(
    hybrid_section: SectionProxy, ao_config: Configuration, ngs_shape: tuple[int, ...]
) -> tuple[u.Quantity, u.Quantity, str | None]:
    """Resolve magnitudes for current INIs and adapt legacy effective rates."""
    flux_key = "ngs_flux_photon_per_s"
    has_magnitudes = "ngs_magnitude" in hybrid_section
    if has_magnitudes == (flux_key in hybrid_section):
        raise ValueError(
            f"Specify exactly one of hybrid.ngs_magnitude and hybrid.{flux_key}."
        )
    key = "ngs_magnitude" if has_magnitudes else flux_key
    values = read_ini_array(hybrid_section, key, label=f"hybrid.{key}")
    if values.shape != ngs_shape:
        raise ValueError(f"hybrid.{key} must match sources_LO coordinates.")
    if has_magnitudes:
        zeropoint = (
            read_ini_scalar(
                hybrid_section,
                "ngs_magnitude_zeropoint",
                label="hybrid.ngs_magnitude_zeropoint",
            )
            * _ZEROPOINT_UNIT
        )
        zeropoint = _positive_scalar(
            zeropoint, _ZEROPOINT_UNIT, "ngs_magnitude_zeropoint", dtype=None
        )
        magnitudes = values * u.mag
        standard = _read_mag_standard(hybrid_section)
    else:
        if np.any(values <= 0.0):
            raise ValueError(f"hybrid.{key} must be positive.")
        zeropoint = _LEGACY_FLUX_ZEROPOINT
        telescope_config = ao_config.telescope
        low_order_detector = ao_config.low_order_detector
        low_order_loop = ao_config.low_order_loop
        if low_order_loop is None or low_order_loop.frame_rate is None:
            raise ValueError("MASTSEL configuration requires RTC.SensorFrameRate_LO.")
        if telescope_config is None or telescope_config.diameter is None:
            raise ValueError("MASTSEL configuration requires telescope.TelescopeDiameter for magnitudes.")
        if low_order_detector is None or low_order_detector.lenslet_counts is None:
            raise ValueError("MASTSEL configuration requires sensor_LO.NumberLenslets for magnitudes.")
        photometry = WFSPhotometryConfig(
            telescope_diameter=telescope_config.diameter,
            n_channels=float(low_order_detector.lenslet_counts[0].value),
            frame_rate=low_order_loop.frame_rate,
            zeropoint=zeropoint,
        )
        magnitudes = photons_per_frame_to_magnitudes(
            values * (u.photon / u.s) / low_order_loop.frame_rate, photometry
        )
        standard = None
    return magnitudes, zeropoint, standard


def _read_mag_standard(section: SectionProxy) -> str | None:
    standard = section.get("ngs_mag_standard")
    if standard is not None and not standard.strip():
        raise ValueError("hybrid.ngs_mag_standard must be nonempty when supplied.")
    return standard


def _decode_calibrated_configuration(
    ini_source: IniSource, ao_config: Configuration, dtype: DTypeLike | None
) -> HybridConfiguration:
    """Parse Cartesian prediction inputs and the shared calibration INI paths."""
    ini_path, base_ini_text, parser = ini_source.path, ini_source.text, ini_source.parser
    assert ini_path is not None and base_ini_text is not None
    hybrid_section = parser[_HYBRID_SECTION]
    allowed = {
        "lo_model",
        "output",
        "science_x_arcsec",
        "science_y_arcsec",
        "ngs_x_arcsec",
        "ngs_y_arcsec",
        "ngs_magnitude",
        "ngs_mag_standard",
        "wavelength_um",
        "zenith_angle_deg",
        "diagnostics_level",
        "non_psd_policy",
    }
    unknown = set(hybrid_section) - allowed
    if unknown:
        raise ValueError(
            f"unsupported E2E [hybrid] keys: {sorted(unknown)}; precision belongs in [e2e]"
        )
    calibration_config = _read_e2e_calibration_configuration(parser, ini_path)
    policy = NonPsdPolicy(hybrid_section.get("non_psd_policy", NonPsdPolicy.ERROR.value))
    if policy is not NonPsdPolicy.ERROR:
        raise ValueError(
            "E2E-calibrated configuration requires non_psd_policy = error."
        )
    request = HybridRequest(
        dtype=calibration_config.dtype if dtype is None else dtype,
        science_positions=_read_cartesian_positions(hybrid_section, "science"),
        ngs_positions=_read_cartesian_positions(hybrid_section, "ngs"),
        ngs_magnitude=read_ini_array(
            hybrid_section, "ngs_magnitude", label="hybrid.ngs_magnitude"
        )
        * u.mag,
        wavelength=read_ini_scalar(
            hybrid_section, "wavelength_um", label="hybrid.wavelength_um"
        )
        * u.um,
        zenith_angle=read_ini_scalar(
            hybrid_section, "zenith_angle_deg", label="hybrid.zenith_angle_deg"
        )
        * u.deg,
        diagnostics_level=hybrid_section.get(
            "diagnostics_level", DiagnosticsLevel.NONE.value
        ),
    )
    if not calibration_config.science_psf.is_file():
        raise FileNotFoundError(
            f"science PSF file not found: {calibration_config.science_psf}"
        )
    return HybridConfiguration.from_config(
        ao_config,
        request=request,
        lo_model=LowOrderModel.E2E_CALIBRATED,
        science_psf_file=calibration_config.science_psf,
        ngs_metric_file=None,
        output=_resolved_path(hybrid_section, "output", source=ini_path, must_exist=False),
        source=ini_path,
        e2e_configuration=ini_path,
        base_config=base_ini_text,
        ngs_mag_standard=_read_mag_standard(hybrid_section),
    )


def _read_cartesian_positions(section, field: str) -> u.Quantity:
    """Read the INI's x and y vectors as a single array of angular pairs."""
    x_key, y_key = f"{field}_x_arcsec", f"{field}_y_arcsec"
    x = read_ini_array(section, x_key, label=f"hybrid.{x_key}")
    y = read_ini_array(section, y_key, label=f"hybrid.{y_key}")
    if x.shape != y.shape:
        raise ValueError(
            f"hybrid.{x_key} and hybrid.{y_key} must have the same length."
        )
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


__all__ = ["HybridConfiguration", "load_configuration"]
