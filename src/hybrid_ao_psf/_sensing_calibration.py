"""Complete calibrated sensing response and covariance-preserving noise shape."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from astropy import units as u
from numpy.typing import DTypeLike

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
from ._sensing_noise import SensingNoiseMeasurements
from ._sensing_response import (
    SensingPrediction,
    SensingResponseModel,
    SensingResponseResult,
)
from ._sensing_statistics import (
    PROJECTION_EDGES,
    RADIAL_EDGES,
    NoiseDisplacementCase,
    assess_noise_blur,
    fit_noise_blur,
    noise_displacement_statistics,
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
class SensingModel:
    """Complete immutable sensing provider, independent of acquisition/measurement files.

    Attributes:
        response: Gain/log-variance tables with measured field/magnitude/ZA ranges,
            photometric convention and fixed numerical controller.
        blur: Covariance-preserving two-Gaussian shape fitted to pooled measurement
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

    def prepare(self, *, dtype: DTypeLike = np.float32) -> SensingModel:
        """Create immutable runtime state at a requested precision.

        Args:
            dtype: Numerical precision, either float32 or float64.

        Returns:
            A separate sensing response and blur model at the requested precision.

        Raises:
            ValueError: If the precision is unsupported or conversion invalidates the calibrated range.
        """
        return SensingModel(
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
        """Return scalar gain and angular per-axis noise variance for each NGS.

        Args:
            positions: Cartesian x/y coordinates with shape ``(NGS, 2)``.
            magnitudes: Calibrated-convention magnitude for each coordinate, or a scalar.
            zenith_angle: Common angular viewing coordinate.

        Returns:
            Dimensionless gains and per-axis variances in square milliarcseconds,
            in input order.

        Raises:
            TypeError: If a physical input lacks its required quantity.
            ValueError: If inputs are invalid or require interpolation outside measured range.
        """
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

        Args:
            positions: Cartesian x/y coordinates with shape ``(NGS, 2)``.
            magnitudes: NGS magnitudes in the calibrated convention.
            zenith_angle: Common angular viewing coordinate.
            reconstructor: Dimensionless ``(2, 2*NGS)`` weights with interleaved x/y columns.

        Returns:
            Stationary x/y command covariance in square milliarcseconds.

        Raises:
            ValueError: If the reconstruction or calibrated response is invalid,
                or the loop is unstable.
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
        model: Saveable sensing provider; sensor tables alone are insufficient.
        diagnostics: Case-balanced optimizer evidence, full radial/projection
            statistics and per-case in-sample distribution/width checks.
    """

    model: SensingModel
    diagnostics: Mapping

    def __post_init__(self):
        if not isinstance(self.model, SensingModel):
            raise TypeError("model must be complete SensingModel")
        object.__setattr__(self, "diagnostics", immutable_metadata(self.diagnostics))


def build_sensing_calibration(
    response: SensingResponseResult,
    noise_measurements: Iterable[SensingNoiseMeasurements],
    *,
    dtype: DTypeLike = np.float32,
) -> SensingCalibrationResult:
    """Fit the complete noise-displacement shape from reusable pooled histories.

    Use every supplied case/frame/realization and retain equal total weight for
    single- and multiple-NGS families. Reject duplicate cases, incompatible
    controllers or missing measured ZA coverage. Shape fitting whitens with the
    pooled empirical covariance; runtime width remains the analytical covariance.
    No source files, detector, wavefronts, native imports or subclass are needed.

    Args:
        response: Response calibration used to simulate the noise histories.
        noise_measurements: Retained cases covering all measured zenith planes.
        dtype: Precision of the final fitted model, either float32 or float64.
            Compatibility is checked at each history's captured precision.

    Returns:
        The complete sensing product and displacement-shape fit diagnostics.

    Raises:
        TypeError: If response or measurement records have unsupported types.
        ValueError: If measurement grid is incomplete, histories conflict with
            the response, or the fit is invalid.
    """
    if not isinstance(response, SensingResponseResult):
        raise TypeError("prepared sensing response/interpolators required")
    dtype = validate_psf_dtype(dtype)
    validation_models = {response.model.dtype: response.model}
    cases = []
    widths = []
    angles = set()
    identities = set()
    for measurement in noise_measurements:
        if not isinstance(measurement, SensingNoiseMeasurements):
            raise TypeError("typed sensing noise_measurements required")
        # Validate against the response as evaluated when this history was captured.
        if measurement.dtype not in validation_models:
            validation_models[measurement.dtype] = response.model.prepare(dtype=measurement.dtype)
        validation_model = validation_models[measurement.dtype]
        controller = validation_model.controller
        if (
            measurement.controller.gain != controller.gain
            or measurement.controller.delay != controller.delay
            or measurement.controller.frame_rate != controller.frame_rate
        ):
            raise ValueError("measurement controller differs from response calibration")
        if any(
            getattr(measurement.settings, key) != getattr(response._sampling, key)
            for key in ("zero_draws", "gain_draws", "block_draws")
        ) or measurement.settings.displacement != response._displacement.astype(
            measurement.dtype
        ):
            raise ValueError("measurement WFS sampling differs from response calibration")
        request = measurement.request
        predicted = validation_model.predict(
            positions=request.positions,
            magnitudes=request.magnitudes,
            zenith_angle=request.zenith_angle,
        )
        if not np.allclose(
            predicted.gain.value, measurement.gains.value, rtol=1e-10, atol=1e-10
        ) or not np.allclose(
            predicted.variance.value, measurement.variances.value, rtol=1e-10, atol=1e-10
        ):
            raise ValueError("measurement gains/variances differ from response calibration")
        angle = float(request.zenith_angle.value)
        zenith_plane = validation_model._measured_zenith_index(request.zenith_angle)
        identity = (zenith_plane, request.key)
        if identity in identities:
            raise ValueError("duplicate case in sensing shape fit")
        identities.add(identity)
        angles.add(zenith_plane)
        statistics = noise_displacement_statistics(
            measurement.displacements.value.transpose(2, 0, 1), dtype=dtype
        )
        case = NoiseDisplacementCase(
            f"{angle:g}:{request.key}",
            len(request.positions),
            request.population,
            statistics,
        )
        cases.append(case)
        analytical = measurement.analytical_covariance.value
        widths.append(
            {
                "case": case.key,
                "trace_relative_difference": float(
                    np.trace(measurement.measured_covariance.value) / np.trace(analytical)
                    - 1
                ),
                "realizations": measurement.settings.noise_realizations,
                "frames": measurement.settings.noise_frames,
            }
        )
    if angles != set(range(len(response.model.zenith_angles))):
        raise ValueError(
            "noise-shape measurement must cover every measured response zenith plane"
        )
    blur, fit = fit_noise_blur(cases, dtype=dtype)
    model = SensingModel(response=response.model.prepare(dtype=dtype), blur=blur)
    diagnostic = {
        "final_fit": fit,
        "distribution_checks": [assess_noise_blur(blur, case) for case in cases],
        "width_checks": widths,
        "displacement_shape_statistics": {
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
        "measured_inputs": response.measured_point_count,
        "asterisms": len(cases),
        "paired_displacement_samples": sum(
            case.statistics.sample_count for case in cases
        ),
        "centring": "pooled",
        "validation": "in-sample fit diagnostics; independent scientific validation remains separate",
    }
    return SensingCalibrationResult(model=model, diagnostics=diagnostic)


def save_sensing_calibration(model: SensingModel, path: str | Path) -> None:
    """Save the complete shared response/shape product, retaining acquisition data.

    Args:
        model: The complete sensing response and displacement shape.
        path: Destination model.json manifest. Create its parent directories if
            needed. Store the numerical array file beside the manifest. Replace
            the manifest only after saving the array file.

    Raises:
        TypeError: If the input is not the expected calibration record.
    """
    if not isinstance(model, SensingModel):
        raise TypeError("complete SensingModel required")
    response = model.response
    save_product(
        path,
        _KIND,
        {
            "units": _UNITS,
            "precision": model.dtype.name,
            "grid_precision": response.grid_precision.name,
            "producer_precision": None
            if response.producer_dtype is None
            else response.producer_dtype.name,
            "controller": controller_header(model.controller),
            "photometry": photometry_header(response.photometry),
            "field_radius_arcsec": float(response.field_radius.value),
            "magnitude_intervals": response.magnitude_intervals.value.tolist(),
            "noise_blur": model.blur.parameters(),
            "centring": "pooled",
        },
        response._arrays,
    )


def load_sensing_calibration(path: str | Path) -> SensingModel:
    """Load the exact shared runtime schema with immutable state and no native files.

    Args:
        path: Path to the saved model.json manifest.

    Returns:
        The complete sensing response and displacement shape reconstructed at its stored precision.

    Raises:
        FileNotFoundError: If a required file is missing.
        ValueError: If its schema or numerical contents are invalid.
    """
    header, arrays = load_product(path, _KIND)
    if (
        (set(header) - {"precision", "producer_precision", "grid_precision"})
        != {
            "schema",
            "version",
            "array_file",
            "array_file_sha256",
            "units",
            "controller",
            "photometry",
            "field_radius_arcsec",
            "magnitude_intervals",
            "noise_blur",
            "centring",
        }
        or header["units"] != _UNITS
        or header["centring"] != "pooled"
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
        grid_precision=header.get("grid_precision"),
    )
    return SensingModel(
        response=response, blur=NoiseBlurModel(**header["noise_blur"], dtype=dtype)
    )
