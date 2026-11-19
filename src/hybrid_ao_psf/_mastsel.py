"""MASTSEL model configuration, runtime, and fixed-gain noise recovery."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from configparser import BasicInterpolation, ConfigParser
from contextlib import ExitStack, contextmanager
from copy import deepcopy
from dataclasses import dataclass, field
from inspect import Parameter, signature
from numbers import Real
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import numpy as np
from ao_core.models.config import parse_ini, read_ini_array, read_ini_scalar
from ao_core.photometry import WFSPhotometryConfig
from astropy import units as u

from ._units import _PhysicalRecord
from .contracts import (
    HybridCtotCorrection,
    HybridMastselStatistics,
    HybridRequest,
    NonPsdPolicy,
    PsfMetricPrediction,
    PsfMetricProvider,
    _positive_scalar,
    _vector_with_length,
)


@dataclass(frozen=True, kw_only=True)
class MastselModel(_PhysicalRecord):
    """MASTSEL inputs for reusable low-order prediction through ``simulate()``.

    Attributes:
        mastsel_ini: Complete MASTSEL INI text. Each simulation applies its
            observation's field coordinates, science wavelength and zenith angle
            to a separate copy of this configuration.
        ngs_metrics: Provider of NGS high-order EE, FWHM and Strehl.
        ngs_magnitude_zeropoint: Pre-aperture photon flux per area and second at
            magnitude zero. Required when the request supplies magnitudes.
        non_psd_policy: Handling of materially non-PSD MASTSEL covariance.
        ctot_correction: Optional field-wide correction applied to baseline
            MASTSEL covariance before conditioning and blur.

    Construction validates the provider, options and INI syntax. It does not
    start MASTSEL. The same model may be used for multiple observations.
    """

    mastsel_ini: str
    ngs_metrics: PsfMetricProvider
    ngs_magnitude_zeropoint: u.Quantity | None = None
    non_psd_policy: NonPsdPolicy | str = NonPsdPolicy.ERROR
    ctot_correction: HybridCtotCorrection | None = None
    _ngs_metric_file: Path | None = field(default=None, repr=False, compare=False)
    _parsed_ini: ConfigParser = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.mastsel_ini, str) or not self.mastsel_ini.strip():
            raise ValueError("mastsel_ini must not be empty.")
        if not isinstance(self.ngs_metrics, PsfMetricProvider):
            raise TypeError("ngs_metrics must implement PsfMetricProvider.")
        if self.ctot_correction is not None and not isinstance(
            self.ctot_correction, HybridCtotCorrection
        ):
            raise TypeError("ctot_correction must be HybridCtotCorrection or None.")
        zeropoint = self.ngs_magnitude_zeropoint
        if zeropoint is not None:
            unit = u.photon / (u.m**2 * u.s)
            _positive_scalar(zeropoint, unit, "ngs_magnitude_zeropoint", dtype=None)
            zeropoint = _positive_scalar(
                zeropoint, zeropoint.unit, "ngs_magnitude_zeropoint", dtype=None
            )
        ini = self.mastsel_ini.strip() + "\n"
        object.__setattr__(self, "mastsel_ini", ini)
        object.__setattr__(self, "ngs_magnitude_zeropoint", zeropoint)
        object.__setattr__(self, "non_psd_policy", NonPsdPolicy(self.non_psd_policy))
        object.__setattr__(self, "_parsed_ini", parse_ini(ini))

    @property
    def _input_files(self) -> dict[str, str]:
        """Report the known NGS metric and correction files."""
        files = (
            dict(self.ctot_correction.input_files)
            if self.ctot_correction is not None
            else {}
        )
        if self._ngs_metric_file is not None:
            files["ngs_metrics"] = str(self._ngs_metric_file)
        return files

    def _copy_ini(self) -> ConfigParser:
        """Supply an independent parser for observation-specific overrides."""
        return deepcopy(self._parsed_ini)


@dataclass(frozen=True, kw_only=True)
class _MastselPrediction:
    """Conditioned covariance and effective inputs for one MASTSEL calculation."""

    ctot: u.Quantity
    ctot_wavefront: u.Quantity
    angle_to_wavefront_scale: u.Quantity
    psd_clipped: np.ndarray
    ngs_metrics: PsfMetricPrediction
    ngs_flux: u.Quantity
    frame_rate: u.Quantity
    effective_ini: str


@dataclass
class _MastselCapture:
    covariance: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
    reconstructor: np.ndarray | None = None


def _mastsel_frame_rate(parser: ConfigParser) -> u.Quantity:
    """Read the positive shared LO sampling rate from the MASTSEL INI."""
    key = "RTC.SensorFrameRate_LO"
    if not parser.has_option("RTC", "SensorFrameRate_LO"):
        raise ValueError(f"MASTSEL configuration requires {key}.")
    try:
        value = float(parser["RTC"]["SensorFrameRate_LO"])
    except ValueError as exc:
        raise ValueError(f"{key} must be a positive finite scalar.") from exc
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{key} must be a positive finite scalar.")
    return value * u.Hz


def _mastsel_loop_settings(parser: ConfigParser) -> tuple[float | None, int | None]:
    """Read fixed-gain settings from the effective MASTSEL configuration."""
    try:
        gain = parser.getfloat("RTC", "LoopGain_LO", fallback=None)
    except ValueError:
        gain = None
    try:
        delay = parser.getint("RTC", "LoopDelaySteps_LO", fallback=None)
    except ValueError:
        delay = None
    return gain, delay


def _mastsel_photometry(
    parser: ConfigParser, frame_rate: u.Quantity, zeropoint: u.Quantity
) -> WFSPhotometryConfig:
    """Read the geometry used for both directions of the magnitude conversion."""
    if not parser.has_option("telescope", "TelescopeDiameter"):
        raise ValueError("MASTSEL configuration requires telescope.TelescopeDiameter for magnitudes.")
    if not parser.has_option("sensor_LO", "NumberLenslets"):
        raise ValueError("MASTSEL configuration requires sensor_LO.NumberLenslets for magnitudes.")
    diameter = read_ini_scalar(
        parser["telescope"], "TelescopeDiameter", label="telescope.TelescopeDiameter"
    )
    lenslets = read_ini_array(
        parser["sensor_LO"], "NumberLenslets", label="sensor_LO.NumberLenslets"
    )
    return WFSPhotometryConfig(
        telescope_diameter=diameter * u.m,
        n_channels=float(lenslets[0]),
        frame_rate=frame_rate,
        zeropoint=zeropoint,
    )


def _cpu_array(value: object) -> np.ndarray:
    """Copy a native NumPy/CuPy numerical return to host memory."""
    if hasattr(value, "get"):
        value = value.get()
    return np.array(value, dtype=float, copy=True)


def _load_mavis_lo() -> Any:
    try:
        from mastsel import MavisLO
    except Exception as exc:  # pragma: no cover - environment failure.
        raise RuntimeError(
            "MASTSEL is not importable; cannot run Hybrid AO PSF."
        ) from exc
    return MavisLO


def _load_mastsel_decoder() -> Callable[[str], object] | None:
    from mastsel import mavisLO

    return getattr(mavisLO, "parse_config_value", None)


class _MastselIniSection(Mapping[str, object]):
    """Decode options on access, as MASTSEL does for a file-backed INI."""

    def __init__(
        self, parser: ConfigParser, name: str, decode: Callable[[str], object]
    ):
        self._parser = parser
        self._options = parser[name]
        self._decode = decode

    def __getitem__(self, key: str) -> object:
        raw = self._options[key]
        if "%" in raw:
            raw = BasicInterpolation().before_get(
                self._parser, self._options.name, key, raw, self._options
            )
        return self._decode(raw)

    def __contains__(self, key: object) -> bool:
        return key in self._options

    def __iter__(self) -> Iterator[str]:
        return iter(self._options)

    def __len__(self) -> int:
        return len(self._options)


@contextmanager
def _open_mastsel(parser: ConfigParser, effective_ini: str) -> Iterator[Any]:
    """Own the native runtime and any compatibility file for one prediction."""
    mavis_lo = _load_mavis_lo()
    decode = _load_mastsel_decoder()
    parameters = signature(mavis_lo).parameters
    accepts_mapping = "config_dict" in parameters or any(
        item.kind == Parameter.VAR_KEYWORD for item in parameters.values()
    )
    with ExitStack() as resources:
        if decode is None or not accepts_mapping:
            # Older supported releases expose only a file-based constructor.
            directory = resources.enter_context(
                TemporaryDirectory(prefix="hybrid_ao_psf_")
            )
            path = Path(directory) / "simulation.ini"
            path.write_text(effective_ini, encoding="utf-8")
            mlo = mavis_lo(directory, path.stem, verbose=False)
        else:
            values = {
                name: _MastselIniSection(parser, name, decode)
                for name in parser.sections()
            }
            mlo = mavis_lo(config_dict=values, verbose=False)
        if getattr(mlo, "error", False):
            raise RuntimeError(
                "MASTSEL failed to load the effective Hybrid configuration."
            )
        yield mlo


def _mastsel_angle_to_wavefront_scale(model: Any) -> u.Quantity:
    """Read the native conversion in nm per mas of image displacement."""
    mas2nm = getattr(model, "mas2nm", None)
    if mas2nm is None:
        raise ValueError("MASTSEL did not expose mas2nm conversion.")
    return _positive_scalar(
        float(mas2nm) * u.nm / u.mas, u.nm / u.mas, "angle_to_wavefront_scale"
    )


def _compute_mastsel_ctot(
    model: Any,
    request: HybridRequest,
    metrics: PsfMetricPrediction,
    lo_frame_rate: u.Quantity,
    ngs_flux: u.Quantity,
) -> u.Quantity:
    """Evaluate baseline Ctot once with the prepared runtime and resolved inputs."""
    ngs_flux = _vector_with_length(
        ngs_flux, u.photon / u.s, "ngs_flux", len(request.ngs_positions), positive=True
    )
    # Keep native work arrays independent of the read-only request.
    science = request.science_positions.to_value(u.arcsec).copy()
    ngs = request.ngs_positions.to_value(u.arcsec).copy()
    ngs_frame_rates = np.full(len(ngs), lo_frame_rate.to_value(u.Hz))
    return (
        np.asarray(
            model.computeTotalResidualMatrix(
                science,
                ngs,
                ngs_flux.to_value(u.photon / u.s),
                ngs_frame_rates,
                metrics.sr.to_value(u.one),
                metrics.ee.to_value(u.one),
                metrics.fwhm.to_value(u.mas),
                aNGS_FWHM_DL_mas=None,
                doAll=True,
            ),
            dtype=float,
        )
        * u.nm**2
    )


@contextmanager
def _capture_mastsel(model: Any) -> Iterator[_MastselCapture]:
    """Wrap only this runtime's two baseline methods and restore their bindings."""
    names = ("computeCovMatrices", "buildReconstuctor2")
    original = {name: getattr(model, name) for name in names}
    bound = {name: model.__dict__[name] for name in names if name in model.__dict__}
    captured = _MastselCapture()

    def covariance(*args: object, **kwargs: object) -> object:
        result = original["computeCovMatrices"](*args, **kwargs)
        if captured.covariance is not None:
            raise RuntimeError("Expected one baseline MASTSEL covariance calculation.")
        captured.covariance = tuple(_cpu_array(value) for value in result)
        return result

    def reconstruction(*args: object, **kwargs: object) -> object:
        result = original["buildReconstuctor2"](*args, **kwargs)
        if captured.reconstructor is not None:
            raise RuntimeError(
                "Expected one baseline MASTSEL reconstructor calculation."
            )
        captured.reconstructor = _cpu_array(result[0])
        return result

    try:
        model.computeCovMatrices = covariance
        model.buildReconstuctor2 = reconstruction
        yield captured
    finally:
        for name in names:
            if name in bound:
                setattr(model, name, bound[name])
            elif name in model.__dict__:
                delattr(model, name)


def _measurement_noise(
    model: Any,
    count: int,
    lo_frame_rate: u.Quantity,
    angle_to_wavefront_scale: u.Quantity,
) -> u.Quantity | None:
    """Recover filtered noise only; retained nr also includes atmospheric servo.

    Reuse baseline bias and centroid variance. The grid, transfer function and
    normalization match MASTSEL's fixed-gain residual calculation, without
    re-evaluating detector moments or atmospheric spectra. Both axes use avar[0],
    as in baseline MASTSEL. Non-fixed gain has no supported noise-only output.
    """
    gain = model.LoopGain_LO
    if not isinstance(gain, Real):
        return None
    if model.noNoise:
        return np.zeros((2 * count, 2 * count)) * u.nm**2
    values = []
    frame_rate = lo_frame_rate.to_value(u.Hz)
    for index in range(count):
        detector_index = index if len(model.NumberLenslets) == count else 0
        variance = float(_cpu_array(model.avar[index][0]))
        variance *= (
            model.PixelScale_LO[detector_index] ** 2 / model.N_sa_tot_LO[detector_index]
        )
        bias = float(_cpu_array(model.bias[index]))
        if not np.isfinite(bias) or bias == 0:
            raise ValueError(
                "MASTSEL sensor bias must be finite and nonzero for noise recovery."
            )
        frequency = np.linspace(0.25, frame_rate / 2, int(2 * frame_rate))
        if frequency.size < 2:
            raise ValueError(
                "MASTSEL noise recovery requires at least two frequency samples."
            )
        effective_gain = bias * gain
        z = np.exp(2j * np.pi * frequency / frame_rate)
        delayed_gain = effective_gain * z ** (-model.loopDelaySteps_LO)
        hn = delayed_gain / (1 - z**-1 + delayed_gain)
        filtered = variance / bias**2 / (frequency.size - 1) * np.sum(abs(hn) ** 2)
        values.extend((filtered, filtered))
    return np.diag(values) * angle_to_wavefront_scale**2 * u.mas**2


def _collect_mastsel_statistics(
    model: Any,
    captured: _MastselCapture,
    request: HybridRequest,
    lo_frame_rate: u.Quantity,
    angle_to_wavefront_scale: u.Quantity,
) -> HybridMastselStatistics:
    """Prepare the baseline covariance and reconstructor data for a correction."""
    if captured.covariance is None or captured.reconstructor is None:
        raise RuntimeError(
            "MASTSEL baseline did not expose covariance and reconstructor returns."
        )
    count, size = len(request.ngs_positions), len(request.science_positions)
    _, cas, css = captured.covariance
    cas = cas.reshape(size, 2, 2 * count)
    center_indices = np.flatnonzero(
        np.all(request.science_positions.to_value(u.arcsec) == 0, axis=1)
    )
    centre = None
    if center_indices.size:
        centre = cas[center_indices[0]]
    elif count > 1:
        ngs = request.ngs_positions.to_value(u.arcsec).copy()
        _, centre, _ = model.computeCovMatrices(np.zeros((1, 2)), ngs)
        centre = _cpu_array(centre)
    return HybridMastselStatistics(
        ngs_covariance=css * u.nm**2,
        science_ngs_covariance=cas * u.nm**2,
        centre_ngs_covariance=None if centre is None else centre * u.nm**2,
        reconstructor=captured.reconstructor.reshape(size, 2, 2 * count),
        measurement_noise=_measurement_noise(
            model, count, lo_frame_rate, angle_to_wavefront_scale
        ),
    )
