"""Standalone calibrated LO covariance and image-displacement composition.

This module consumes fitted products only. Native acquisition, calibration
workflows, MASTSEL and final science-PSF convolution are separate operations.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from functools import partial
from pathlib import Path
from types import MappingProxyType

import numpy as np
from astropy import units as u
from numpy.typing import DTypeLike, NDArray

from ._calibration_config import load_e2e_calibration_configuration
from ._calibration_contracts import _angle, _quantity
from ._calibration_io import (
    CalibrationArtifactIdentity,
    CalibrationFileIdentity,
    read_calibration_artifact_identity,
)
from ._coma_calibration import ComaResponseModel, load_coma_calibration
from ._covariance_calibration import CovarianceFunctions, load_covariance_calibration
from ._e2e_prediction import E2ELowOrderPrediction
from ._lo_model import (
    _ControlSolution,
    _LowOrderModel,
    _ReconstructorCovariances,
    _TiltCommandCovariances,
)
from ._reconstructor_calibration import (
    ReconstructorNoiseModel,
    _reconstructor_atmospheric_covariances,
    load_reconstructor_calibration,
)
from ._sensing_calibration import SensingCalibration, load_sensing_calibration
from ._validation import validate_psf_dtype
from .interpolation import NgsHoMetricInterpolator, load_ngs_ho_metric_interpolator


@dataclass(frozen=True, kw_only=True, eq=False)
class E2ECalibratedModel(_LowOrderModel):
    """Prepared fitted products for standalone LO displacement prediction.

    Construct from loaded shared products, or use load_e2e_calibrated_model for
    a configured, content-bound model. Construction prepares immutable providers
    once at dtype, including nested state, and rejects controller/photometry
    conflicts. Prediction uses the four fitted calibration products and the
    prepared NGS HO metric provider.

    Attributes:
        covariance: Six calibrated covariance functions and fixed controller.
        coma: Primary-coma response h and saved photometry convention.
        sensing: Scalar gains, per-axis variances and noise-mixture shape.
        reconstructor: Separate calibrated pre-filter noise prior.
        ngs_metrics: Immutable NGS HO EE/FWHM/Strehl interpolator.
        dtype: Runtime arithmetic, default float32 or explicit float64.
        product_identities: Read-only content identities, populated by the INI
            loader for the four products and NGS metric file. Direct construction
            may leave this empty; it makes no file-provenance claim.
    """

    covariance: CovarianceFunctions
    coma: ComaResponseModel
    sensing: SensingCalibration
    reconstructor: ReconstructorNoiseModel
    ngs_metrics: NgsHoMetricInterpolator
    dtype: DTypeLike = np.float32
    product_identities: Mapping[
        str, CalibrationArtifactIdentity | CalibrationFileIdentity
    ] = field(default_factory=dict)

    def __post_init__(self):
        dtype = validate_psf_dtype(self.dtype)
        object.__setattr__(self, "dtype", dtype)
        for name, cls in (
            ("covariance", CovarianceFunctions),
            ("coma", ComaResponseModel),
            ("sensing", SensingCalibration),
            ("reconstructor", ReconstructorNoiseModel),
            ("ngs_metrics", NgsHoMetricInterpolator),
        ):
            value = getattr(self, name)
            if not isinstance(value, cls):
                raise TypeError(f"{name} must be {cls.__name__}")
            object.__setattr__(
                self,
                name,
                value.prepare(dtype=dtype),
            )
        covariance_controller = self.covariance.controller
        sensing_controller = self.sensing.controller
        if (
            covariance_controller.delay != sensing_controller.delay
            or covariance_controller.gain != sensing_controller.gain
            or covariance_controller.frame_rate != sensing_controller.frame_rate
        ):
            raise ValueError("covariance and sensing controller settings differ")
        coma_photometry = self.coma.photometry
        sensing_photometry = self.sensing.response.photometry
        if (
            coma_photometry.magnitude_band != sensing_photometry.magnitude_band
            or coma_photometry.photons_per_frame_zeropoint
            != sensing_photometry.photons_per_frame_zeropoint
        ):
            raise ValueError("coma and sensing photometry conventions differ")
        identities = dict(self.product_identities)
        if identities and set(identities) != {
            "covariance",
            "coma",
            "sensing",
            "reconstructor",
            "ngs_metrics",
        }:
            raise ValueError("product identities must bind all five runtime inputs")
        for name, value in identities.items():
            expected = (
                CalibrationFileIdentity
                if name == "ngs_metrics"
                else CalibrationArtifactIdentity
            )
            if not isinstance(value, expected):
                raise TypeError(f"invalid {name} product identity")
        object.__setattr__(self, "product_identities", MappingProxyType(identities))

    def prepare(self, *, dtype: DTypeLike) -> E2ECalibratedModel:
        """Return immutable runtime state at dtype without altering saved products.

        Args:
            dtype: Numerical precision, either float32 or float64.

        Returns:
            A separate prepared model with the accepted product identities at the requested precision.

        Raises:
            ValueError: If the precision is unsupported or conversion invalidates support.
        """
        dtype = validate_psf_dtype(dtype)
        return self if dtype == self.dtype else replace(self, dtype=dtype)

    def predict(
        self,
        *,
        ngs_positions: u.Quantity,
        magnitudes: u.Quantity,
        science_positions: u.Quantity,
        zenith_angle: u.Quantity,
    ) -> E2ELowOrderPrediction:
        """Compose atmospheric, HO and noise covariances at physical field points.

        NGS and science positions have shape (N,2) and (S,2), in angular units;
        magnitudes is a length-N vector in the saved photometry convention.
        zenith_angle is scalar. Empty/duplicate NGS, nonphysical inputs,
        unsupported interpolation, unstable/defective feedback and non-PSD
        noiseless results raise. No spatial, magnitude, ZA or feedback
        extrapolation or fallback to another physical model is performed.

        Args:
            ngs_positions: Cartesian angular NGS coordinates of shape ``(NGS, 2)``.
            magnitudes: One magnitude per NGS in the calibrated convention.
            science_positions: Cartesian angular query coordinates of shape
                ``(science, 2)``.
            zenith_angle: Common scalar observing angle.

        Returns:
            A self-contained low-order prediction with component covariances and
            the applied Gaussian mixture for each science position.

        Raises:
            TypeError: If required inputs are not physical quantities.
            ValueError: If geometry, photometry, fitted support or low-order
                covariance and feedback are invalid.
        """
        dtype = self.dtype
        ngs_positions = _quantity(
            ngs_positions, u.arcsec, "ngs_positions", 2, dtype=dtype
        )
        science_positions = _quantity(
            science_positions, u.arcsec, "science_positions", 2, dtype=dtype
        )
        magnitudes = _quantity(magnitudes, u.mag, "magnitudes", 1, dtype=dtype)
        zenith_angle = _angle(zenith_angle, dtype=dtype)
        if (
            ngs_positions.shape != (len(magnitudes), 2)
            or not len(ngs_positions)
            or len(np.unique(ngs_positions.value, axis=0)) != len(ngs_positions)
        ):
            raise ValueError("distinct NGS positions must align with magnitudes")
        if science_positions.shape[1:] != (2,) or not len(science_positions):
            raise ValueError("science_positions must be nonempty (S,2)")

        # Evaluate the fitted inputs for sensing, coma, and reconstruction.
        sensing = self.sensing.predict(
            positions=ngs_positions, magnitudes=magnitudes, zenith_angle=zenith_angle
        )
        coma_responses = self.coma.predict(
            magnitude=magnitudes, zenith_angle=zenith_angle
        )
        metrics = self.ngs_metrics.predict(
            zenith_angle=zenith_angle, x=ngs_positions[:, 0], y=ngs_positions[:, 1]
        )
        photons_per_frame = self.coma.photometry.photons_per_frame(magnitudes)
        reconstructor_noise_prior = self.reconstructor.measurement_noise(
            photons_per_frame=photons_per_frame, ee=metrics.ee, fwhm=metrics.fwhm
        )
        reconstruction_covariances = None
        if len(ngs_positions) > 1:
            ngs_covariance, centre_ngs_covariance = (
                _reconstructor_atmospheric_covariances(
                    self.covariance, ngs_positions, zenith_angle
                )
            )
            reconstruction_covariances = _ReconstructorCovariances(
                ngs_covariance=ngs_covariance,
                centre_ngs_covariance=centre_ngs_covariance,
                noise_prior=reconstructor_noise_prior,
            )

        # The shared solve supplies the control response to this evaluator.
        evaluate_command_covariances = partial(
            _calibrated_command_covariances,
            self.covariance,
            ngs_positions,
            science_positions,
            sensing.gain.value,
            coma_responses.value,
            zenith_angle,
        )
        prediction = self._predict_low_order(
            reconstruction_covariances=reconstruction_covariances,
            sensing_gains=sensing.gain.value,
            wfs_noise_variances=sensing.variance.value,
            controller=self.covariance.controller,
            evaluate_covariances=evaluate_command_covariances,
        )

        control, terms = prediction.control, prediction.terms
        scalar_feedback = control.scalar_feedback

        # Each Gaussian combines the noiseless residual with a scaled noise covariance.
        narrow, broad = self.sensing.blur.scales(len(ngs_positions))
        scales = np.array([narrow, broad], dtype=dtype)
        weight = self.sensing.blur.mixture_weight
        mixture = (
            prediction.noiseless_covariance.value[:, None]
            + scales[None, :, None, None] * prediction.noise_command_covariance.value
        )
        return E2ELowOrderPrediction._create(
            dtype,
            photons_per_frame=photons_per_frame,
            metrics=metrics,
            coma_response=coma_responses,
            gains=sensing.gain,
            noise_variances=sensing.variance,
            noise_prior=reconstructor_noise_prior,
            reconstructor=control.reconstructor,
            feedback=control.feedback,
            effective_feedback=scalar_feedback.effective_feedback,
            strengths=scalar_feedback.strengths,
            projectors=scalar_feedback.projectors,
            symmetric_feedback_fallback=scalar_feedback.symmetric_fallback,
            feedback_approximation_relative_error=scalar_feedback.approximation_relative_error,
            eigenbasis_condition=scalar_feedback.eigenbasis_condition,
            atmospheric_command=terms.atmospheric_command * u.mas**2,
            ho_command=terms.ho_command * u.mas**2,
            atmospheric_ho_command=terms.atmospheric_ho_command * u.mas**2,
            science_atmospheric_command=terms.science_atmospheric_command * u.mas**2,
            science_ho_command=terms.science_ho_command * u.mas**2,
            uncorrected=terms.science_tilt * u.mas**2,
            atmospheric=prediction.atmospheric_residual_covariance,
            ho=prediction.ho_covariance_contribution,
            noiseless=prediction.noiseless_covariance,
            noise=prediction.noise_command_covariance,
            total=prediction.total_covariance,
            mixture_weights=np.array([weight, 1 - weight], dtype=dtype),
            noise_scales=scales,
            mixture_covariances=mixture * u.mas**2,
        )


def _calibrated_command_covariances(
    covariance: CovarianceFunctions,
    ngs_positions: u.Quantity,
    science_positions: u.Quantity,
    sensing_gains: NDArray,
    coma_responses: NDArray,
    zenith_angle: u.Quantity,
    control: _ControlSolution,
) -> _TiltCommandCovariances:
    """Contract the six calibrated functions with the scalar closed-loop response.

    Borrow the fitted bank, physical field coordinates, and numerical WFS
    responses for one asterism. Use the solved reconstructor and scalar
    feedback projectors to return science and command covariances in mas².

    Preserve the order of the two projector sums. Each P_p multiplies its
    reconstructor block from the left. The mixed command term has atmosphere
    on the left and HO on the right.
    """
    dtype = control.reconstructor.dtype
    scalar_feedback = control.scalar_feedback
    strengths = scalar_feedback.strengths
    projected_reconstructors = np.einsum(
        "pab,gbc->gpac", scalar_feedback.projectors, control.reconstructor_blocks
    )
    atmospheric_command = np.zeros((2, 2), dtype=dtype)
    ho_command = np.zeros_like(atmospheric_command)
    atmospheric_ho_command = np.zeros_like(atmospheric_command)
    science_atmospheric_command = np.zeros((len(science_positions), 2, 2), dtype=dtype)
    science_ho_command = np.zeros_like(science_atmospheric_command)
    science_x, science_y = science_positions.T
    for i, (xi, yi) in enumerate(ngs_positions):
        for p, b1 in enumerate(strengths):
            left = projected_reconstructors[i, p]
            science_query = (science_x, science_y, xi, yi, b1)
            science_atmospheric_command += (
                sensing_gains[i]
                * covariance.C_t_Tt(*science_query, zenith_angle=zenith_angle).value
                @ left.T
            )
            science_ho_command += (
                coma_responses[i]
                * covariance.C_t_Tc(*science_query, zenith_angle=zenith_angle).value
                @ left.T
            )
            for g, (xg, yg) in enumerate(ngs_positions):
                for q, b2 in enumerate(strengths):
                    right = projected_reconstructors[g, q].T
                    ngs_pair_query = (xi, yi, xg, yg, b1, b2)
                    atmospheric_command += (
                        sensing_gains[i]
                        * sensing_gains[g]
                        * left
                        @ covariance.C_Tt_Tt(
                            *ngs_pair_query, zenith_angle=zenith_angle
                        ).value
                        @ right
                    )
                    ho_command += (
                        coma_responses[i]
                        * coma_responses[g]
                        * left
                        @ covariance.C_Tc_Tc(
                            *ngs_pair_query, zenith_angle=zenith_angle
                        ).value
                        @ right
                    )
                    atmospheric_ho_command += (
                        sensing_gains[i]
                        * coma_responses[g]
                        * left
                        @ covariance.C_Tt_Tc(
                            *ngs_pair_query, zenith_angle=zenith_angle
                        ).value
                        @ right
                    )
    science_tilt = covariance.C_t_t(
        science_x, science_y, science_x, science_y, zenith_angle=zenith_angle
    ).value
    return _TiltCommandCovariances(
        science_tilt=science_tilt,
        atmospheric_command=atmospheric_command,
        ho_command=ho_command,
        atmospheric_ho_command=atmospheric_ho_command,
        science_atmospheric_command=science_atmospheric_command,
        science_ho_command=science_ho_command,
    )


def load_e2e_calibrated_model(
    path: str | Path, *, dtype: DTypeLike | None = None
) -> E2ECalibratedModel:
    """Load and prepare complete calibrated LO state from a shared E2E INI.

    Resolve defaults then INI precision then explicit dtype. Open only the four
    model.json products and the NGS metric interpolator; native sections,
    acquisition sources, science PSFs and NGS spot PSFs are not opened.
    Each fitted loader validates its schema/support/conventions. Product-content
    identities bind the loaded bytes and are available for orchestration reuse.
    Missing/incompatible products raise; no native calibration is started.

    Args:
        path: Shared INI locating the four products and the NGS metric provider.
        dtype: Optional float32 or float64 override of the configured precision.

    Returns:
        A prepared low-order model with validated input identities.

    Raises:
        FileNotFoundError: If the configuration or a provider file is missing.
        ValueError: If products conflict or their data fail validation.
    """
    config = load_e2e_calibration_configuration(path)
    paths = {
        name: getattr(config, name + "_model")
        for name in ("covariance", "coma", "sensing", "reconstructor")
    }
    identities = {
        name: read_calibration_artifact_identity(value) for name, value in paths.items()
    }
    identities["ngs_metrics"] = CalibrationFileIdentity(
        config.ngs_metrics, hashlib.sha256(config.ngs_metrics.read_bytes()).hexdigest()
    )
    model = E2ECalibratedModel(
        covariance=load_covariance_calibration(paths["covariance"]),
        coma=load_coma_calibration(paths["coma"]),
        sensing=load_sensing_calibration(paths["sensing"]),
        reconstructor=load_reconstructor_calibration(paths["reconstructor"]),
        ngs_metrics=load_ngs_ho_metric_interpolator(config.ngs_metrics),
        dtype=config.dtype if dtype is None else dtype,
        product_identities=identities,
    )
    for identity in identities.values():
        identity.verify()
    return model
