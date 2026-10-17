"""Complete calibrated sensing response and covariance-preserving noise shape."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from astropy import units as u

from ._calibration_contracts import FixedGainIntegrator
from ._calibration_products import (
    controller_header,
    immutable_metadata,
    load_product,
    photometry_header,
    read_controller_header,
    read_photometry_header,
    save_product,
)
from ._noise_blur import NoiseBlurModel
from ._sensing_replay import SensingReplayMeasurements
from ._sensing_response import (
    SensingPrediction,
    SensingResponseCalibration,
    SensingResponseModel,
)
from ._sensing_statistics import (
    PROJECTION_EDGES,
    RADIAL_EDGES,
    ReplayCase,
    assess_noise_blur,
    fit_noise_blur,
    replay_statistics,
)
from ._validation import validate_psf_dtype

_KIND = "hybrid-ao-psf-sensing-calibration"
_UNITS = {
    "position": "arcsec",
    "magnitude": "mag",
    "zenith": "deg",
    "variance": "mas2",
    "gain": "",
}


@dataclass(frozen=True, kw_only=True)
class SensingCalibration:
    """Complete immutable sensing provider, independent of acquisition/replay files.

    Attributes:
        response: Gain/log-variance tables with measured field/magnitude/ZA support,
            photometric convention and fixed numerical controller.
        blur: Covariance-preserving two-Gaussian shape fitted to pooled replay
            histories with equal single-/multiple-NGS family weights.

    Runtime width is calculated with the actual feedback matrix and the
    Lyapunov equation. Empirical shape fitting introduces no second noise scale.
    """

    response: SensingResponseModel
    blur: NoiseBlurModel

    def __post_init__(self):
        if not isinstance(self.response, SensingResponseModel) or not isinstance(
            self.blur, NoiseBlurModel
        ):
            raise TypeError("typed complete sensing response and noise shape required")

    @property
    def dtype(self) -> np.dtype:
        """Numerical precision used by this complete runtime model."""
        return self.response.dtype

    def prepare(self, *, dtype=np.float32) -> SensingCalibration:
        """Create immutable runtime state at a requested precision."""
        return SensingCalibration(
            response=self.response.prepare(dtype=dtype),
            blur=self.blur.prepare(dtype=dtype),
        )

    @property
    def controller(self) -> FixedGainIntegrator:
        """Return the fixed gain, delay and frame rate declared by this calibration."""
        return self.response.controller

    def predict(
        self, *, positions: u.Quantity, magnitudes: u.Quantity, zenith_angle: u.Quantity
    ) -> SensingPrediction:
        """Return scalar gain and angular per-axis noise variance for each NGS."""
        return self.response.predict(
            positions=positions, magnitudes=magnitudes, zenith_angle=zenith_angle
        )

    def noise_covariance(
        self,
        *,
        positions: u.Quantity,
        magnitudes: u.Quantity,
        zenith_angle: u.Quantity,
        reconstructor: np.ndarray,
    ) -> u.Quantity:
        """Calculate stationary noise-command covariance for supplied absolute weights.

        The reconstructor is finite dimensionless (2,2*N). Use the actual B=sum
        g_g*R_g and reconstructed white-noise covariance sum v_g*R_g*R_g.T,
        without a symmetric feedback approximation. Return a 2-by-2 mas-squared
        covariance. Gain/delay/cadence come from this accepted calibration.
        """
        values = self.predict(
            positions=positions, magnitudes=magnitudes, zenith_angle=zenith_angle
        )
        weights = np.asarray(reconstructor, dtype=self.dtype)
        if (
            weights.shape != (2, 2 * len(values.gain))
            or weights.dtype.kind not in "fiu"
            or not np.isfinite(weights).all()
        ):
            raise ValueError("finite two-by-2N dimensionless reconstructor required")
        blocks = weights.reshape(2, len(values.gain), 2).transpose(1, 0, 2)
        feedback = np.einsum("g,gab->ab", values.gain.value, blocks)
        noise = np.einsum("g,gab,gcb->ac", values.variance.value, blocks, blocks)
        return self.controller.noise_covariance(feedback, noise * u.mas**2)


@dataclass(frozen=True, kw_only=True)
class SensingCalibrationResult:
    """Complete model plus immutable interpolation/shape/width-fit diagnostics.

    Attributes:
        model: Publishable sensing provider; sensor tables alone are insufficient.
        diagnostics: Case-balanced optimizer evidence, full radial/projection
            statistics and per-case in-sample distribution/width checks.
    """

    model: SensingCalibration
    diagnostics: Mapping

    def __post_init__(self):
        if not isinstance(self.model, SensingCalibration):
            raise TypeError("model must be complete SensingCalibration")
        object.__setattr__(self, "diagnostics", immutable_metadata(self.diagnostics))


def build_sensing_calibration(
    response: SensingResponseCalibration,
    replays: Iterable[SensingReplayMeasurements],
    *,
    dtype=np.float32,
) -> SensingCalibrationResult:
    """Fit the complete noise-displacement shape from reusable pooled histories.

    Use every supplied case/frame/realization and retain equal total weight for
    single- and multiple-NGS families. Reject duplicate cases, incompatible
    controllers or missing measured ZA coverage. Shape fitting whitens with the
    pooled empirical covariance; runtime width remains the analytical covariance.
    No source files, detector, wavefronts, native imports or subclass are needed.
    """
    if not isinstance(response, SensingResponseCalibration):
        raise TypeError("prepared sensing response/banks required")
    dtype = validate_psf_dtype(dtype)
    response = response.prepare(dtype=dtype)
    cases = []
    widths = []
    angles = set()
    identities = set()
    controller = response.model.controller
    for replay in replays:
        if not isinstance(replay, SensingReplayMeasurements):
            raise TypeError("typed sensing replays required")
        if (
            replay.controller.gain != controller.gain
            or replay.controller.delay != controller.delay
            or replay.controller.frame_rate != controller.frame_rate
        ):
            raise ValueError("replay controller differs from response calibration")
        if (
            any(
                getattr(replay.settings, key) != getattr(response._sampling, key)
                for key in ("zero_draws", "gain_draws", "block_draws")
            )
            or replay.settings.displacement != response._displacement
        ):
            raise ValueError("replay WFS sampling differs from response calibration")
        request = replay.request
        predicted = response.model.predict(
            positions=request.positions,
            magnitudes=request.magnitudes,
            zenith_angle=request.zenith_angle,
        )
        if not np.allclose(
            predicted.gain.value, replay.gains.value, rtol=1e-10, atol=1e-10
        ) or not np.allclose(
            predicted.variance.value, replay.variances.value, rtol=1e-10, atol=1e-10
        ):
            raise ValueError("replay gains/variances differ from response calibration")
        angle = float(request.zenith_angle.value)
        identity = (angle, request.key)
        if identity in identities:
            raise ValueError("duplicate case in sensing shape fit")
        identities.add(identity)
        angles.add(angle)
        statistics = replay_statistics(
            replay.displacements.value.transpose(2, 0, 1), dtype=dtype
        )
        case = ReplayCase(
            f"{angle:g}:{request.key}",
            len(request.positions),
            request.population,
            statistics,
        )
        cases.append(case)
        analytical = replay.analytical_covariance.value
        widths.append(
            {
                "case": case.key,
                "trace_relative_difference": float(
                    np.trace(replay.measured_covariance.value) / np.trace(analytical)
                    - 1
                ),
                "realizations": replay.settings.replay_realizations,
                "frames": replay.settings.replay_frames,
            }
        )
    if angles != set(map(float, response.model.zenith_angles.value)):
        raise ValueError(
            "noise-shape replay must cover every measured response zenith plane"
        )
    blur, fit = fit_noise_blur(cases, dtype=dtype)
    model = SensingCalibration(response=response.model, blur=blur)
    diagnostic = {
        "final_fit": fit,
        "distribution_checks": [assess_noise_blur(blur, case) for case in cases],
        "width_checks": widths,
        "replay_shape_statistics": {
            "case_names": [case.key for case in cases],
            "radial_edges": RADIAL_EDGES[:-1].tolist(),
            "radial_infinite_tail": True,
            "projection_edges": PROJECTION_EDGES.tolist(),
            "radial_probabilities": [
                case.statistics.radial_probability.tolist() for case in cases
            ],
            "projection_cdf": [
                case.statistics.projection_cdf.tolist() for case in cases
            ],
        },
        "measured_inputs": response.bank_count,
        "asterisms": len(cases),
        "paired_displacement_samples": sum(
            case.statistics.sample_count for case in cases
        ),
        "replay_centring": "pooled",
        "validation": "in-sample fit diagnostics; independent scientific validation remains separate",
    }
    return SensingCalibrationResult(model=model, diagnostics=diagnostic)


def save_sensing_calibration(model: SensingCalibration, path: str | Path) -> None:
    """Publish the complete shared response/shape product, retaining acquisition data."""
    if not isinstance(model, SensingCalibration):
        raise TypeError("complete SensingCalibration required")
    response = model.response
    save_product(
        path,
        _KIND,
        {
            "units": _UNITS,
            "precision": model.dtype.name,
            "support_precision": response.support_precision.name,
            "producer_precision": None
            if response.producer_dtype is None
            else response.producer_dtype.name,
            "controller": controller_header(model.controller),
            "photometry": photometry_header(response.photometry),
            "field_radius_arcsec": float(response.field_radius.value),
            "magnitude_intervals": response.magnitude_intervals.value.tolist(),
            "noise_blur": model.blur.parameters(),
            "replay_centring": "pooled",
        },
        response._arrays,
    )


def load_sensing_calibration(path: str | Path) -> SensingCalibration:
    """Load the exact shared runtime schema with immutable state and no native files."""
    header, arrays = load_product(path, _KIND)
    if (
        (set(header) - {"precision", "producer_precision", "support_precision"})
        != {
            "schema",
            "version",
            "payload",
            "payload_sha256",
            "units",
            "controller",
            "photometry",
            "field_radius_arcsec",
            "magnitude_intervals",
            "noise_blur",
            "replay_centring",
        }
        or header["units"] != _UNITS
        or header["replay_centring"] != "pooled"
        or arrays is None
        or set(header["noise_blur"])
        != {"mixture_weight", "mixture_single_scale", "mixture_multi_scale"}
    ):
        raise ValueError("invalid sensing model fields, units or centring")
    dtype = validate_psf_dtype(header.get("precision", arrays["gain"].dtype))
    response = SensingResponseModel._from_arrays(
        arrays,
        read_controller_header(header["controller"], dtype=dtype),
        read_photometry_header(header["photometry"], dtype=dtype),
        header["field_radius_arcsec"] * u.arcsec,
        header["magnitude_intervals"],
        producer_dtype=header.get("producer_precision"),
        support_precision=header.get("support_precision"),
    )
    return SensingCalibration(
        response=response, blur=NoiseBlurModel(**header["noise_blur"], dtype=dtype)
    )
