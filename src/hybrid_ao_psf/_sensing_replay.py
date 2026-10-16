"""Offline delayed-loop replay of paired WFS noise with calibrated LO weights."""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from astropy import units as u

from ._calibration_config import SensingCalibrationSettings
from ._calibration_contracts import (
    FixedGainIntegrator,
    _angle,
    _count,
    _PhysicalRecord,
    _quantity,
)
from ._calibration_numerics import filter_input
from ._calibration_products import controller_header, read_controller_header
from ._covariance_calibration import CovarianceFunctions
from ._io import atomic_output_path
from ._reconstructor_calibration import ReconstructorNoiseModel, build_lo_reconstructor
from ._sensing_response import SensingResponseCalibration
from ._sensing_statistics import pooled_replay_moments

_KIND = "hybrid-ao-psf-sensing-replay"
_ARRAY_UNITS = {
    "positions": "arcsec",
    "magnitudes": "mag",
    "ee": "",
    "fwhm": "mas",
    "displacements": "mas",
    "reconstructor": "",
    "feedback": "",
    "gains": "",
    "variances": "mas2",
    "noise_prior": "mas2",
    "analytical_covariance": "mas2",
    "measured_covariance": "mas2",
    "pooled_mean": "mas",
    "realization_means": "mas",
}


@dataclass(frozen=True, kw_only=True, eq=False)
class SensingReplayRequest(_PhysicalRecord):
    """One synthetic asterism and its explicit replay inputs, without native state.

    Attributes:
        key: Nonempty case label, unique with zenith angle within a fit.
        population: Caller-defined sampling label retained in diagnostics.
        zenith_angle: Scalar physical viewing angle of the captured noise banks.
        positions: Cartesian NGS directions, shape (N,2), angular.
        magnitudes: Per-NGS magnitudes in the measured photometric convention.
        ee: Per-NGS dimensionless HO encircled-energy fractions in (0,1].
        fwhm: Per-NGS positive angular HO full widths at half maximum.
        seed: Nonnegative explicit NumPy generator seed for this case.

    Photon counts come from the captured photometric conversion, never MASTSEL
    or a native setup. A recipe may resolve EE/FWHM from its configured provider
    before exporting these reusable inputs.
    """

    key: str
    population: str
    zenith_angle: u.Quantity
    positions: u.Quantity
    magnitudes: u.Quantity
    ee: u.Quantity
    fwhm: u.Quantity
    seed: int

    def __post_init__(self):
        if any(
            not isinstance(value, str) or not value.strip()
            for value in (self.key, self.population)
        ):
            raise ValueError("nonempty replay key and population required")
        _count(self.seed, "seed", 0)
        values = {
            "zenith_angle": _angle(self.zenith_angle),
            "positions": _quantity(self.positions, u.arcsec, "positions", 2),
            "magnitudes": _quantity(self.magnitudes, u.mag, "magnitudes", 1),
            "ee": _quantity(self.ee, u.one, "ee", 1),
            "fwhm": _quantity(self.fwhm, u.mas, "fwhm", 1),
        }
        n = len(values["positions"])
        if (
            values["positions"].shape != (n, 2)
            or any(values[key].shape != (n,) for key in ("magnitudes", "ee", "fwhm"))
            or np.any(values["ee"].value <= 0)
            or np.any(values["ee"].value > 1)
            or np.any(values["fwhm"].value <= 0)
        ):
            raise ValueError(
                "aligned physical NGS positions/magnitudes/EE/FWHM required"
            )
        for key, value in values.items():
            object.__setattr__(self, key, value)


@dataclass(frozen=True, kw_only=True, eq=False)
class SensingReplayMeasurements(_PhysicalRecord):
    """One retained command-displacement history with reproducible numeric inputs.

    Attributes:
        request: Case inputs and explicit generator seed.
        controller: Fixed numerical delayed integrator used for this case.
        settings: Realization, retained-frame and settling-discard controls.
        displacements: Pooled-centred (realizations,frames,2) angular samples,
            at the retained float32 serialization precision.
        reconstructor: Dimensionless absolute LO weights, shape (2,2*N).
        feedback: Actual reconstructed sensing response, shape (2,2).
        gains: Dimensionless scalar local gains, shape (N,).
        variances: Per-NGS measured angular-squared noise variances.
        noise_prior: Separate pre-filter prior for reconstruction, shape (2*N,2*N).
        analytical_covariance: Lyapunov command covariance, shape (2,2), angular squared.
        measured_covariance: Replay population covariance before float32 storage.
        pooled_mean: Two-component mean removed across retained frames/realizations.
        realization_means: Per-realization angular means before pooled centring.

    Between-realization offsets remain in the distribution. These histories
    calibrate shape; the analytical covariance supplies runtime width. Retained
    histories must agree with their pre-storage covariance and means within
    float32 quantization and float64 reduction bounds.
    """

    request: SensingReplayRequest
    controller: FixedGainIntegrator
    settings: SensingCalibrationSettings
    displacements: u.Quantity
    reconstructor: u.Quantity
    feedback: u.Quantity
    gains: u.Quantity
    variances: u.Quantity
    noise_prior: u.Quantity
    analytical_covariance: u.Quantity
    measured_covariance: u.Quantity
    pooled_mean: u.Quantity
    realization_means: u.Quantity

    def __post_init__(self):
        if (
            not isinstance(self.request, SensingReplayRequest)
            or not isinstance(self.controller, FixedGainIntegrator)
            or not isinstance(self.settings, SensingCalibrationSettings)
        ):
            raise TypeError("typed replay request/controller/settings required")
        n = len(self.request.positions)
        r = self.settings.replay_realizations
        f = self.settings.replay_frames
        specs = {
            "displacements": (u.mas, (r, f, 2)),
            "reconstructor": (u.one, (2, 2 * n)),
            "feedback": (u.one, (2, 2)),
            "gains": (u.one, (n,)),
            "variances": (u.mas**2, (n,)),
            "noise_prior": (u.mas**2, (2 * n, 2 * n)),
            "analytical_covariance": (u.mas**2, (2, 2)),
            "measured_covariance": (u.mas**2, (2, 2)),
            "pooled_mean": (u.mas, (2,)),
            "realization_means": (u.mas, (r, 2)),
        }
        for name, (unit, shape) in specs.items():
            value = _quantity(getattr(self, name), unit, name, len(shape))
            if value.shape != shape:
                raise ValueError(f"incorrect replay {name} shape")
            object.__setattr__(self, name, value)
        if np.any(self.gains.value <= 0) or np.any(self.variances.value <= 0):
            raise ValueError("positive replay gains/variances required")
        for name in ("noise_prior", "analytical_covariance", "measured_covariance"):
            covariance = getattr(self, name).value
            if not np.allclose(
                covariance, covariance.T, rtol=1e-12, atol=1e-12
            ) or np.linalg.eigvalsh(
                (covariance + covariance.T) / 2
            ).min() < -1e-12 * max(1.0, np.linalg.norm(covariance)):
                raise ValueError(
                    f"replay {name} must be symmetric positive semidefinite"
                )
        blocks = self.reconstructor.value.reshape(2, n, 2).transpose(1, 0, 2)
        feedback = np.einsum("g,gab->ab", self.gains.value, blocks)
        if not np.allclose(feedback, self.feedback.value, rtol=1e-12, atol=1e-12):
            raise ValueError("replay feedback differs from declared weights/gains")
        innovation = np.einsum("g,gab,gcb->ac", self.variances.value, blocks, blocks)
        covariance = self.controller.noise_covariance(feedback, innovation * u.mas**2)
        if not np.allclose(
            covariance.value, self.analytical_covariance.value, rtol=1e-10, atol=1e-10
        ):
            raise ValueError(
                "replay analytical covariance differs from declared controller/noise"
            )

        _validate_history_moments(self)


def _validate_history_moments(record: SensingReplayMeasurements) -> None:
    """Check pre-storage moments against pooled-centred, float32-rounded histories.

    Quantization bounds use half an IEEE float32 ulp relative to each retained
    component, including subnormals. Float64 reduction/centring roundoff is
    bounded separately. This does not compare empirical and stationary widths
    or require individual realization means to vanish.
    """
    samples = record.displacements.value
    count = samples.shape[0] * samples.shape[1]
    flat = samples.reshape(-1, 2)
    second = flat.T @ flat / count
    rms = np.sqrt(np.diag(second))
    unit_roundoff = np.finfo(np.float32).eps / 2
    relative_error = unit_roundoff / (1 - unit_roundoff)
    absolute_error = float(np.finfo(np.float32).smallest_subnormal)
    error_rms = relative_error * rms + absolute_error
    reduction_error = (count + 2) * np.finfo(float).eps
    covariance_bound = (
        np.outer(rms, error_rms)
        + np.outer(error_rms, rms)
        + np.outer(error_rms, error_rms)
        + reduction_error * np.outer(rms, rms)
    )
    if np.any(np.abs(second - record.measured_covariance.value) > covariance_bound):
        raise ValueError("replay history moments differ from measured covariance")
    original_means = record.realization_means.value
    pooled = record.pooled_mean.value
    mean_scale = np.maximum(np.max(np.abs(original_means), axis=0), np.abs(pooled))
    numerical_bound = (
        reduction_error * np.maximum(mean_scale, rms) + np.finfo(float).tiny
    )
    if np.any(np.abs(original_means.mean(0) - pooled) > numerical_bound):
        raise ValueError(
            "replay history moments have inconsistent pooled/realization means"
        )
    quantization_bound = relative_error * np.abs(samples).mean(axis=1) + absolute_error
    if np.any(
        np.abs(samples.mean(axis=1) - (original_means - pooled))
        > quantization_bound + numerical_bound
    ):
        raise ValueError(
            "replay history moments differ from recorded realization means"
        )


def replay_sensing_calibration(
    response: SensingResponseCalibration,
    requests: Iterable[SensingReplayRequest],
    covariances: CovarianceFunctions,
    noise_prior: ReconstructorNoiseModel,
    *,
    settings: SensingCalibrationSettings | None = None,
) -> Iterator[SensingReplayMeasurements]:
    """Replay cases independently with captured paired banks and calibrated weights.

    Use zero command prehistory, discard the declared settling interval, then
    remove one pooled mean across retained frames and realizations. Preserve
    the original star/draw order, rectangular bank mixing and float32 retained
    histories. No E2E modules/files or calibration object are needed.
    """
    if (
        not isinstance(response, SensingResponseCalibration)
        or not isinstance(covariances, CovarianceFunctions)
        or not isinstance(noise_prior, ReconstructorNoiseModel)
    ):
        raise TypeError("prepared sensing response/covariance/noise models required")
    settings = SensingCalibrationSettings() if settings is None else settings
    if not isinstance(settings, SensingCalibrationSettings):
        raise TypeError("typed sensing settings required")
    if (
        any(
            getattr(settings, key) != getattr(response._sampling, key)
            for key in ("zero_draws", "gain_draws", "block_draws")
        )
        or settings.displacement != response._displacement
    ):
        raise ValueError(
            "replay sampling/displacement settings differ from captured observations"
        )
    controller = response.model.controller
    if (
        controller.gain != covariances.controller.gain
        or controller.delay != covariances.controller.delay
        or controller.frame_rate != covariances.controller.frame_rate
    ):
        raise ValueError("sensing and covariance controllers differ")
    seen = set()
    for request in requests:
        if not isinstance(request, SensingReplayRequest):
            raise TypeError("typed replay requests required")
        identity = (float(request.zenith_angle.value), request.key)
        if identity in seen:
            raise ValueError("duplicate replay case")
        seen.add(identity)
        values = response.model.predict(
            positions=request.positions,
            magnitudes=request.magnitudes,
            zenith_angle=request.zenith_angle,
        )
        photons = response.model.photometry.photons_per_frame(request.magnitudes)
        prior = noise_prior.measurement_noise(
            photons_per_frame=photons, ee=request.ee, fwhm=request.fwhm
        )
        weights = build_lo_reconstructor(
            covariances,
            noise_prior,
            ngs_positions=request.positions,
            zenith_angle=request.zenith_angle,
            photons_per_frame=photons,
            ee=request.ee,
            fwhm=request.fwhm,
        )
        n = len(request.positions)
        blocks = weights.reshape(2, n, 2).transpose(1, 0, 2)
        gains, variance = values.gain.value, values.variance.value
        feedback = np.einsum("g,gab->ab", gains, blocks)
        reconstructed_noise = np.einsum("g,gab,gcb->ac", variance, blocks, blocks)
        expected = controller.noise_covariance(feedback, reconstructed_noise * u.mas**2)
        frames = settings.replay_frames + settings.discard_frames
        realizations = settings.replay_realizations
        x = np.zeros((2, realizations, frames))
        rng = np.random.default_rng(request.seed)
        for g in range(n):
            samples = response.draw_noise(
                position=request.positions[g],
                magnitude=request.magnitudes[g],
                zenith_angle=request.zenith_angle,
                rng=rng,
                count=realizations * frames,
            )
            samples *= np.sqrt(variance[g])
            values = samples.T.reshape(2, realizations, frames)
            x += np.einsum("ab,brt->art", blocks[g], values)
        displacements = -filter_input(x, feedback, controller.gain, controller.delay)[
            ..., settings.discard_frames :
        ]
        realization_means = displacements.mean(-1)
        displacements, mean, covariance = pooled_replay_moments(displacements)
        yield SensingReplayMeasurements(
            request=request,
            controller=controller,
            settings=settings,
            displacements=displacements.astype(np.float32).transpose(1, 2, 0) * u.mas,
            reconstructor=weights * u.one,
            feedback=feedback * u.one,
            gains=gains * u.one,
            variances=variance * u.mas**2,
            noise_prior=prior,
            analytical_covariance=expected,
            measured_covariance=covariance * u.mas**2,
            pooled_mean=mean * u.mas,
            realization_means=realization_means.T * u.mas,
        )


def _settings_header(settings):
    return {
        **{
            name: value
            for name, value in settings.__dict__.items()
            if name != "displacement"
        },
        "displacement_mas": float(settings.displacement.value),
    }


def save_sensing_replay(replay: SensingReplayMeasurements, path: str | Path) -> None:
    """Atomically retain one full replay result and its physical reconstruction inputs."""
    if not isinstance(replay, SensingReplayMeasurements):
        raise TypeError("unsupported sensing replay type")
    request = replay.request
    header = {
        "schema": _KIND,
        "version": 1,
        "units": _ARRAY_UNITS,
        "centring": "pooled",
        "key": request.key,
        "population": request.population,
        "zenith_angle_deg": float(request.zenith_angle.value),
        "seed": request.seed,
        "controller": controller_header(replay.controller),
        "settings": _settings_header(replay.settings),
    }
    arrays = {
        name: getattr(request, name).value
        if name in ("positions", "magnitudes", "ee", "fwhm")
        else getattr(replay, name).value
        for name in _ARRAY_UNITS
    }
    # The common physical record stores exact float32 values as float64;
    # persist at the accepted captured precision rather than invent new bits.
    arrays["displacements"] = arrays["displacements"].astype(np.float32)
    with atomic_output_path(Path(path)) as temporary, temporary.open("wb") as stream:
        np.savez_compressed(
            stream, __header__=json.dumps(header, allow_nan=False), **arrays
        )


def load_sensing_replay(path: str | Path) -> SensingReplayMeasurements:
    """Load exactly the shared pooled-replay schema, never native provenance files."""
    with np.load(Path(path), allow_pickle=False) as saved:
        header = json.loads(str(saved["__header__"]))
        if (
            set(header)
            != {
                "schema",
                "version",
                "units",
                "centring",
                "key",
                "population",
                "zenith_angle_deg",
                "seed",
                "controller",
                "settings",
            }
            or header["schema"] != _KIND
            or type(header["version"]) is not int
            or header["version"] != 1
            or header["units"] != _ARRAY_UNITS
            or header["centring"] != "pooled"
            or set(saved.files) != {"__header__", *_ARRAY_UNITS}
        ):
            raise ValueError("invalid sensing replay schema, units, centring or fields")
        settings = dict(header["settings"])
        settings["displacement"] = settings.pop("displacement_mas") * u.mas
        request = SensingReplayRequest(
            key=header["key"],
            population=header["population"],
            zenith_angle=header["zenith_angle_deg"] * u.deg,
            seed=header["seed"],
            **{
                name: saved[name] * u.Unit(_ARRAY_UNITS[name])
                for name in ("positions", "magnitudes", "ee", "fwhm")
            },
        )
        return SensingReplayMeasurements(
            request=request,
            controller=read_controller_header(header["controller"]),
            settings=SensingCalibrationSettings(**settings),
            **{
                name: saved[name] * u.Unit(unit)
                for name, unit in _ARRAY_UNITS.items()
                if name not in ("positions", "magnitudes", "ee", "fwhm")
            },
        )
