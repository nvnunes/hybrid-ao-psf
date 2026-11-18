"""Shared E2E calibration configuration, without native setup interpretation."""

from __future__ import annotations

import ast
from collections.abc import Mapping
from configparser import ConfigParser
from dataclasses import dataclass, field, fields, replace
from pathlib import Path

import numpy as np
from astropy import units as u
from numpy.typing import DTypeLike

from ._calibration_contracts import (
    SensingWfsSampling,
    _count,
    _PhysicalRecord,
    _quantity,
)
from ._validation import validate_psf_dtype


@dataclass(frozen=True, kw_only=True)
class CovarianceCalibrationSettings:
    """Scalar interpolator controls; fixed controller settings come from the adapter.

    Attributes:
        feedback: Increasing positive dimensionless scalar feedback grid points.
        settling_tolerance: Positive bound on retained initial-state propagation.
        maximum_discard: Maximum saved-frame settling cut to consider.
    """

    feedback: tuple[float, ...] = tuple(n / 10 for n in range(2, 21))
    settling_tolerance: float = 1e-8
    maximum_discard: int = 500

    def __post_init__(self):
        values = np.asarray(self.feedback)
        if (
            values.ndim != 1
            or len(values) < 2
            or values.dtype.kind not in "fiu"
            or not np.isfinite(values).all()
            or np.any(values <= 0)
            or np.any(np.diff(values) <= 0)
        ):
            raise ValueError("feedback values must be increasing, positive and finite")
        object.__setattr__(self, "feedback", tuple(map(float, values)))
        if (
            not np.isfinite(self.settling_tolerance)
            or not 0 < self.settling_tolerance < 1
        ):
            raise ValueError("settling_tolerance must lie in (0,1)")
        _count(self.maximum_discard, "maximum_discard")


@dataclass(frozen=True, kw_only=True)
class ComaCalibrationSettings:
    """Wavefront and detector-noise sampling settings.

    Attributes:
        samples: Independent wavefronts, a positive multiple of four.
        noise_draws: Draws per wavefront/magnitude, a multiple of four.
        noise_seed: Native uint32 seed.
    """

    samples: int = 512
    noise_draws: int = 512
    noise_seed: int = 10003000

    def __post_init__(self):
        for name in ("samples", "noise_draws"):
            _count(getattr(self, name), name, 4)
            if getattr(self, name) % 4:
                raise ValueError(f"{name} must be divisible by four")
        _count(self.noise_seed, "noise_seed", 0)
        if self.noise_seed >= 2**32:
            raise ValueError("noise_seed must fit uint32")


@dataclass(frozen=True, kw_only=True)
class SensingCalibrationSettings(_PhysicalRecord):
    """Fixed-spot draw and offline command-simulation controls.

    Attributes:
        zero_draws: Draws for conditional noise statistics.
        gain_draws: Draws per signed displacement.
        block_draws: Equal native batch length.
        seed: Native draw/simulation seed, below 2**31.
        noise_realizations: Independent command histories per case.
        noise_frames: Retained simulation frames after settling.
        discard_frames: Initial simulation frames excluded from statistics.
        displacement: Positive scalar angular imposed displacement amplitude.
    """

    zero_draws: int = 4096
    gain_draws: int = 16384
    block_draws: int = 1024
    seed: int = 261013700
    noise_realizations: int = 8
    noise_frames: int = 8192
    discard_frames: int = 512
    displacement: u.Quantity = field(default_factory=lambda: 10.0 * u.mas)

    def __post_init__(self):
        SensingWfsSampling(
            zero_draws=self.zero_draws,
            gain_draws=self.gain_draws,
            block_draws=self.block_draws,
            seed=self.seed,
        )
        for name in ("noise_realizations", "noise_frames", "discard_frames"):
            _count(getattr(self, name), name)
        displacement = _quantity(self.displacement, u.mas, "displacement", 0)
        if self.noise_frames < 4 or displacement.value <= 0:
            raise ValueError("need >=4 simulation frames and positive finite displacement")
        object.__setattr__(self, "displacement", displacement)


@dataclass(frozen=True, kw_only=True)
class ReconstructorCalibrationSettings:
    """Native-prior table/build controls, without detector configuration.

    Attributes:
        representation: Product P*EE or separate P,EE table axes.
        signal_point_count: Number of points along the photon or P*EE axis.
        fwhm_point_count: Number of points along the FWHM axis.
        ee_point_count: Number of points along the EE axis for the separate representation.
        padding: Fractional range padding, in [0,1).
        interpolation: Linear or log-linear covariance interpolation.
    """

    representation: str = "product"
    signal_point_count: int = 129
    fwhm_point_count: int = 97
    ee_point_count: int = 17
    padding: float = 0.1
    interpolation: str = "log-linear"

    def __post_init__(self):
        if self.representation not in (
            "product",
            "separate",
        ) or self.interpolation not in ("linear", "log-linear"):
            raise ValueError("unsupported noise-prior representation/interpolation")
        for name in ("signal_point_count", "fwhm_point_count", "ee_point_count"):
            _count(getattr(self, name), name, 2)
        if not np.isfinite(self.padding) or not 0 <= self.padding < 1:
            raise ValueError("padding must lie in [0,1)")


@dataclass(frozen=True, kw_only=True)
class E2ECalibrationConfiguration:
    """Shared INI fields and effective controls, with no native dependencies.

    Attributes:
        dtype: Resolved real calculation precision, float32 by default.
        source: Absolute shared INI path.
        covariance_model: Planned or existing covariance model.json path.
        coma_model: Planned or existing response model.json path.
        sensing_model: Planned or existing sensor model.json path.
        reconstructor_model: Planned or existing prior model.json path.
        science_psf: Science HO interpolator path.
        ngs_psf: NGS HO PSF interpolator path for fixed-spot measurements.
        ngs_metrics: NGS HO metric interpolator path for reconstruction.
        covariance: Effective common interpolator controls.
        coma: Effective response controls.
        sensing: Effective fixed-spot/simulation controls.
        reconstructor: Effective prior controls.

    Paths are resolved relative to the INI, without opening them. Loading this
    configuration does not imply that calibration or source products exist.
    """

    dtype: DTypeLike = np.float32
    source: Path
    covariance_model: Path
    coma_model: Path
    sensing_model: Path
    reconstructor_model: Path
    science_psf: Path
    ngs_psf: Path
    ngs_metrics: Path
    covariance: CovarianceCalibrationSettings = field(
        default_factory=CovarianceCalibrationSettings
    )
    coma: ComaCalibrationSettings = field(default_factory=ComaCalibrationSettings)
    sensing: SensingCalibrationSettings = field(
        default_factory=SensingCalibrationSettings
    )
    reconstructor: ReconstructorCalibrationSettings = field(
        default_factory=ReconstructorCalibrationSettings
    )

    def __post_init__(self):
        object.__setattr__(self, "dtype", validate_psf_dtype(self.dtype))
        object.__setattr__(self, "source", Path(self.source).expanduser().resolve())
        for name in (
            "covariance_model",
            "coma_model",
            "sensing_model",
            "reconstructor_model",
            "science_psf",
            "ngs_psf",
            "ngs_metrics",
        ):
            path = Path(getattr(self, name)).expanduser()
            if not path.is_absolute():
                path = self.source.parent / path
            object.__setattr__(self, name, path.resolve())
        targets = [getattr(self, family + "_model") for family in _SETTINGS]
        if len(set(targets)) != len(targets):
            raise ValueError("calibration model destinations must be distinct")
        for name, cls in _SETTINGS.items():
            if not isinstance(getattr(self, name), cls):
                raise TypeError(f"{name} must be {cls.__name__}")
        for name in (
            "covariance_model",
            "coma_model",
            "sensing_model",
            "reconstructor_model",
        ):
            if getattr(self, name).name != "model.json":
                raise ValueError(f"{name} must reference a model.json entrypoint")

    def with_overrides(
        self,
        *,
        dtype: DTypeLike | None = None,
        covariance: Mapping[str, object] | None = None,
        coma: Mapping[str, object] | None = None,
        sensing: Mapping[str, object] | None = None,
        reconstructor: Mapping[str, object] | None = None,
    ) -> E2ECalibrationConfiguration:
        """Apply explicit mapping overrides above INI/defaults, returning a new config.

        Omit an option to retain its existing value. Unknown fields and explicit
        invalid values raise; callers should omit absent CLI options entirely.

        Args:
            dtype: Optional calculation precision override.
            covariance: Overrides for named covariance settings.
            coma: Overrides for named coma settings.
            sensing: Overrides for named sensing settings.
            reconstructor: Overrides for named reconstructor settings.

        Returns:
            A new validated configuration with the same input and output paths.

        Raises:
            ValueError: If an override changes a field to an unsupported value.
            TypeError: If an override names an unsupported setting.
        """
        updates = {} if dtype is None else {"dtype": validate_psf_dtype(dtype)}
        for name, override in [
            ("covariance", covariance),
            ("coma", coma),
            ("sensing", sensing),
            ("reconstructor", reconstructor),
        ]:
            if override is not None:
                updates[name] = replace(getattr(self, name), **override)
        return replace(self, **updates)


_SETTINGS = {
    "covariance": CovarianceCalibrationSettings,
    "coma": ComaCalibrationSettings,
    "sensing": SensingCalibrationSettings,
    "reconstructor": ReconstructorCalibrationSettings,
}
_MODEL_KEYS = (
    "covariance_model",
    "coma_model",
    "sensing_model",
    "reconstructor_model",
    "science_psf",
    "ngs_psf",
    "ngs_metrics",
)


def load_e2e_calibration_configuration(path: str | Path) -> E2ECalibrationConfiguration:
    """Read shared [e2e] and [calibration.<family>] sections of one INI.

    Required [e2e] paths are resolved relative to the INI without opening model
    or native sources. Other sections belong to downstream owners. Unknown keys
    in owned sections raise. Values use Python literals; enum strings may be
    unquoted. Fixed controller and photometry settings are supplied by the
    configured native adapter, then saved with accepted products.

    Args:
        path: Shared calibration INI. Relative file paths resolve from its directory.

    Returns:
        Effective settings and resolved file paths for calibration and prediction.

    Raises:
        FileNotFoundError: If the INI is missing.
        ValueError: If required keys are missing or owned settings are invalid.
    """
    source = Path(path).expanduser().resolve()
    parser = ConfigParser(interpolation=None)
    parser.optionxform = str
    with source.open(encoding="utf-8") as stream:
        parser.read_file(stream)
    unknown_sections = {
        section for section in parser.sections() if section.startswith("calibration.")
    } - {f"calibration.{name}" for name in _SETTINGS}
    if unknown_sections:
        raise ValueError(f"Unknown calibration sections: {sorted(unknown_sections)}")
    if parser.defaults():
        raise ValueError("[DEFAULT] values are not supported in the shared E2E INI")
    if not parser.has_section("e2e") or (set(parser["e2e"]) - {"precision"}) != set(
        _MODEL_KEYS
    ):
        raise ValueError(f"[e2e] requires exactly {', '.join(_MODEL_KEYS)}")
    values = {"dtype": validate_psf_dtype(parser["e2e"].get("precision", "float32"))}
    for key in _MODEL_KEYS:
        raw = parser["e2e"][key].strip()
        if not raw:
            raise ValueError(f"[e2e] {key} must not be empty")
        p = Path(raw).expanduser()
        values[key] = (
            (source.parent / p).resolve() if not p.is_absolute() else p.resolve()
        )
    for family, cls in _SETTINGS.items():
        section = f"calibration.{family}"
        raw_values = dict(parser[section]) if parser.has_section(section) else {}
        unknown = raw_values.keys() - {item.name for item in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown keys in [{section}]: {sorted(unknown)}")
        kwargs = {}
        for name, raw in raw_values.items():
            if isinstance(getattr(cls(), name), u.Quantity):
                kwargs[name] = u.Quantity(raw)
                continue
            try:
                kwargs[name] = ast.literal_eval(raw)
            except (ValueError, SyntaxError):
                if isinstance(getattr(cls(), name), str):
                    kwargs[name] = raw
                else:
                    raise TypeError(f"Invalid [{section}] {name}: {raw}") from None
        values[family] = cls(**kwargs)
    return E2ECalibrationConfiguration(source=source, **values)
