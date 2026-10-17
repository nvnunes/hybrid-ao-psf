"""Standalone calibrated LO covariance and image-displacement composition.

This module consumes fitted products only. Native acquisition, calibration
workflows, MASTSEL and final science-PSF convolution are separate operations.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from types import MappingProxyType

import numpy as np
from astropy import units as u
from numpy.typing import DTypeLike

from ._calibration_config import load_e2e_calibration_configuration
from ._calibration_contracts import _angle, _PhysicalRecord, _quantity
from ._calibration_io import (
    CalibrationArtifactIdentity,
    CalibrationFileIdentity,
    read_calibration_artifact_identity,
)
from ._coma_calibration import ComaResponseModel, load_coma_calibration
from ._covariance_calibration import CovarianceFunctions, load_covariance_calibration
from ._reconstructor_calibration import (
    ReconstructorNoiseModel,
    build_lo_reconstructor,
    load_reconstructor_calibration,
)
from ._sensing_calibration import SensingCalibration, load_sensing_calibration
from ._validation import validate_psf_dtype
from .contracts import NgsHoMetricPrediction
from .interpolation import NgsHoMetricInterpolator, load_ngs_ho_metric_interpolator


def _array(value, dtype):
    value = np.ascontiguousarray(value, dtype=dtype)
    if not np.isfinite(value).all():
        raise ValueError("calibrated prediction contains nonfinite values")
    return np.frombuffer(value.tobytes(), dtype=value.dtype).reshape(value.shape)


@dataclass(frozen=True, init=False, eq=False)
class E2ELowOrderPrediction(_PhysicalRecord):
    """Immutable scientific result from E2ECalibratedModel.predict.

    All matrices use physical x/y axes. Angular covariances are in mas squared.
    Results are created by the model, not by a caller-supplied constructor.

    Attributes:
        dtype: Selected runtime float32 or float64 arithmetic.
        photons_per_frame: Incident per-NGS photon counts, (N,).
        metrics: Prepared NGS EE, FWHM and Strehl vectors.
        coma_response: Per-NGS h, (N,), in mas per nm OPD.
        gains: Per-NGS dimensionless g, (N,).
        noise_variances: Per-NGS per-axis v, (N,), in mas squared.
        noise_prior: Separate pre-filter reconstructor prior, (2N,2N).
        reconstructor: Absolute field-centre LO weights, (2,2N).
        feedback: Actual B=sum(g_g R_g), (2,2).
        effective_feedback: B used for the scalar covariance bank, (2,2).
        strengths: Two real eigenvalues of effective_feedback.
        projectors: Corresponding, generally non-orthogonal (2,2,2) projections.
        symmetric_feedback_fallback: Whether the complex-eigenvalue symmetric
            approximation was needed for bank evaluation only.
        feedback_approximation_relative_error: Relative Frobenius change in B.
        eigenbasis_condition: Condition number of the chosen eigenvector basis.
        atmospheric_command: Cov(u_atm), (2,2).
        ho_command: Cov(u_HO), (2,2).
        atmospheric_ho_command: Cov(u_atm,u_HO), (2,2); not generally symmetric.
        science_atmospheric_command: Cov(t_science,u_atm), (S,2,2).
        science_ho_command: Cov(t_science,u_HO), (S,2,2).
        uncorrected: Atmospheric science-tilt covariance, (S,2,2).
        atmospheric: Atmospheric residual covariance, (S,2,2).
        ho: HO correction including both cross terms, (S,2,2); may be indefinite.
        noiseless: Atmospheric plus HO covariance, (S,2,2).
        noise: Stationary Lyapunov noise-command covariance using actual B, (2,2).
        total: Noiseless plus noise covariance, (S,2,2).
        mixture_weights: Narrow/broad noise-mixture weights, (2,).
        noise_scales: Covariance-preserving narrow/broad scales, (2,).
        mixture_covariances: Full displacement Gaussian component covariances,
            (S,2,2,2), ordered science, mixture component, x, y.

    No per-component PSD clipping is applied. The model rejects a noiseless
    covariance with an eigenvalue below -1e-8 mas squared; tiny roundoff is
    retained explicitly rather than hidden by conditioning.
    """

    dtype: np.dtype
    photons_per_frame: u.Quantity
    metrics: NgsHoMetricPrediction
    coma_response: u.Quantity
    gains: u.Quantity
    noise_variances: u.Quantity
    noise_prior: u.Quantity
    reconstructor: np.ndarray
    feedback: np.ndarray
    effective_feedback: np.ndarray
    strengths: np.ndarray
    projectors: np.ndarray
    symmetric_feedback_fallback: bool
    feedback_approximation_relative_error: float
    eigenbasis_condition: float
    atmospheric_command: u.Quantity
    ho_command: u.Quantity
    atmospheric_ho_command: u.Quantity
    science_atmospheric_command: u.Quantity
    science_ho_command: u.Quantity
    uncorrected: u.Quantity
    atmospheric: u.Quantity
    ho: u.Quantity
    noiseless: u.Quantity
    noise: u.Quantity
    total: u.Quantity
    mixture_weights: np.ndarray
    noise_scales: np.ndarray
    mixture_covariances: u.Quantity

    def __init__(self):
        raise TypeError("Use E2ECalibratedModel.predict")

    @classmethod
    def _create(cls, dtype, **values):
        """Validate self-contained prediction state for runtime and I/O.

        Check feedback/reconstruction, scalar decomposition, covariance priors
        and displacement-mixture identities without modifying stored values.
        """
        dtype = validate_psf_dtype(dtype)
        expected = {item.name for item in fields(cls)} - {"dtype"}
        if set(values) != expected:
            raise ValueError(
                "prediction state must contain exactly its declared fields"
            )
        result = object.__new__(cls)
        object.__setattr__(result, "dtype", dtype)
        units = {
            "photons_per_frame": u.photon,
            "coma_response": u.mas / u.nm,
            "gains": u.one,
            "noise_variances": u.mas**2,
            "noise_prior": u.mas**2,
            **{
                name: u.mas**2
                for name in (
                    "atmospheric_command",
                    "ho_command",
                    "atmospheric_ho_command",
                    "science_atmospheric_command",
                    "science_ho_command",
                    "uncorrected",
                    "atmospheric",
                    "ho",
                    "noiseless",
                    "noise",
                    "total",
                    "mixture_covariances",
                )
            },
        }
        for name, value in values.items():
            if name in units:
                if not isinstance(value, u.Quantity):
                    raise TypeError(f"{name} must be a quantity")
                value = _quantity(value, units[name], name, value.ndim, dtype=dtype)
            elif name == "metrics":
                if not isinstance(value, NgsHoMetricPrediction):
                    raise TypeError("metrics must be NgsHoMetricPrediction")
                value = replace(value, dtype=dtype)
            elif name == "symmetric_feedback_fallback":
                if not isinstance(value, (bool, np.bool_)):
                    raise TypeError("feedback fallback must be boolean")
                value = bool(value)
            elif name in (
                "feedback_approximation_relative_error",
                "eigenbasis_condition",
            ):
                if np.ndim(value) != 0 or not np.isfinite(value) or value < 0:
                    raise ValueError(f"invalid {name}")
                value = float(value)
            else:
                if np.asarray(value).dtype.kind not in "fiu":
                    raise ValueError(f"{name} must be real")
                value = _array(value, dtype)
            object.__setattr__(result, name, value)
        n = len(result.metrics.ee)
        if result.uncorrected.ndim != 3:
            raise ValueError("uncorrected covariance must be (science,2,2)")
        count = len(result.uncorrected)
        shapes = {
            **{
                key: (n,)
                for key in (
                    "photons_per_frame",
                    "coma_response",
                    "gains",
                    "noise_variances",
                )
            },
            "noise_prior": (2 * n, 2 * n),
            "reconstructor": (2, 2 * n),
            "feedback": (2, 2),
            "effective_feedback": (2, 2),
            "strengths": (2,),
            "projectors": (2, 2, 2),
            "mixture_weights": (2,),
            "noise_scales": (2,),
            **{
                key: (2, 2)
                for key in (
                    "atmospheric_command",
                    "ho_command",
                    "atmospheric_ho_command",
                    "noise",
                )
            },
            **{
                key: (count, 2, 2)
                for key in (
                    "science_atmospheric_command",
                    "science_ho_command",
                    "uncorrected",
                    "atmospheric",
                    "ho",
                    "noiseless",
                    "total",
                )
            },
            "mixture_covariances": (count, 2, 2, 2),
        }
        for name, shape in shapes.items():
            if getattr(result, name).shape != shape:
                raise ValueError(f"prediction {name} must have shape {shape}")
        if any(
            np.any(getattr(result, key).value <= 0)
            for key in (
                "photons_per_frame",
                "coma_response",
                "gains",
                "noise_variances",
            )
        ):
            raise ValueError(
                "positive photon counts and WFS response/noise values required"
            )
        tolerance = max(1e-10, 32 * np.finfo(dtype).eps)
        blocks = result.reconstructor.reshape(2, n, 2).transpose(1, 0, 2)
        if not np.allclose(
            result.feedback,
            np.einsum("g,gab->ab", result.gains.value, blocks),
            rtol=tolerance,
            atol=tolerance,
        ):
            raise ValueError("feedback must equal the gain-weighted reconstructor sum")
        _validate_scalar_decomposition(
            result.feedback,
            result.effective_feedback,
            result.strengths,
            result.projectors,
            result.symmetric_feedback_fallback,
        )
        if not np.allclose(
            result.noiseless.value,
            result.atmospheric.value + result.ho.value,
            rtol=tolerance,
            atol=1e-8,
        ):
            raise ValueError("noiseless covariance must equal atmospheric plus HO")
        if not np.allclose(
            result.total.value,
            result.noiseless.value + result.noise.value,
            rtol=tolerance,
            atol=1e-8,
        ):
            raise ValueError("total covariance must equal noiseless plus noise")
        if np.any(result.mixture_weights <= 0) or not np.isclose(
            result.mixture_weights.sum(), 1, rtol=tolerance
        ):
            raise ValueError("positive mixture weights must sum to one")
        if np.any(result.noise_scales <= 0) or not np.isclose(
            result.mixture_weights @ result.noise_scales, 1, rtol=tolerance
        ):
            raise ValueError("noise mixture must preserve covariance")
        expected = (
            result.noiseless.value[:, None]
            + result.noise_scales[None, :, None, None] * result.noise.value
        )
        if not np.allclose(
            result.mixture_covariances.value, expected, rtol=tolerance, atol=1e-8
        ):
            raise ValueError(
                "mixture covariances differ from noiseless and scaled noise"
            )
        for name in (
            "noise_prior",
            "uncorrected",
            "atmospheric",
            "noiseless",
            "noise",
            "total",
            "mixture_covariances",
        ):
            value = getattr(result, name).value
            if (
                not np.allclose(
                    value, value.swapaxes(-1, -2), rtol=tolerance, atol=1e-8
                )
                or np.linalg.eigvalsh(value).min() < -1e-8
            ):
                raise ValueError(
                    f"prediction {name} must be symmetric positive semidefinite"
                )
        return result


def _validate_scalar_decomposition(
    feedback, effective, strengths, projectors, fallback
):
    """Check the scalar representation shared by runtime prediction and loading.

    The effective matrix must equal the actual feedback, or its symmetric part
    when the fallback is declared. Projectors must be idempotent, resolve identity
    and reproduce that matrix with the stored strengths. The idempotence check
    allows two-term matrix-product roundoff for non-orthogonal projectors.
    Values are checked, never conditioned.
    """
    tolerance = max(1e-10, 32 * np.finfo(feedback.dtype).eps)
    expected = (feedback + feedback.T) / 2 if fallback else feedback
    if not np.allclose(effective, expected, rtol=tolerance, atol=tolerance):
        raise ValueError("effective feedback must match the declared feedback response")
    if not np.allclose(
        projectors.sum(0), np.eye(2), rtol=tolerance, atol=tolerance
    ) or not np.allclose(
        np.einsum("p,pab->ab", strengths, projectors),
        effective,
        rtol=tolerance,
        atol=tolerance,
    ):
        raise ValueError("feedback scalar decomposition failed reconstruction")
    eps = np.finfo(feedback.dtype).eps
    # Each 2x2 product entry is a two-term dot product, with error bounded
    # componentwise by gamma_2 * (abs(P) @ abs(P)).
    gamma_2 = 2 * eps / (1 - 2 * eps)
    with np.errstate(over="ignore", invalid="ignore"):
        products = projectors @ projectors
        product_roundoff = gamma_2 * (np.abs(projectors) @ np.abs(projectors))
        allowed = tolerance + tolerance * np.abs(projectors) + product_roundoff
    if (
        not np.isfinite(products).all()
        or not np.isfinite(allowed).all()
        or not np.all(np.abs(products - projectors) <= allowed)
    ):
        raise ValueError("feedback scalar decomposition requires idempotent projectors")


def _scalar_decomposition(feedback):
    """Resolve real projectors, preserving the accepted complex-B approximation.

    A defective or excessively ill-conditioned eigenbasis cannot represent the
    calibrated scalar bank. B itself remains unchanged for noise propagation.
    """
    dtype = feedback.dtype
    eps = np.finfo(dtype).eps
    values, basis = np.linalg.eig(feedback)
    fallback = bool(np.max(np.abs(values.imag)) > 1e-10)
    effective = (feedback + feedback.T) / 2 if fallback else feedback
    if fallback:
        values, basis = np.linalg.eigh(effective)
    else:
        values, basis = values.real, basis.real
    condition = float(np.linalg.cond(basis))
    if condition > min(1e10, 1 / (100 * eps)):
        if np.allclose(
            effective,
            np.eye(2, dtype=dtype) * np.trace(effective) / 2,
            rtol=max(1e-10, 4 * eps),
            atol=max(1e-12, eps),
        ):
            basis = np.eye(2, dtype=dtype)
            values = np.repeat(np.trace(effective) / 2, 2)
            condition = 1.0
        else:
            raise ValueError("feedback has an ill-conditioned or defective eigenbasis")
    inverse = np.linalg.inv(basis)
    projections = np.stack([np.outer(basis[:, p], inverse[p]) for p in range(2)])
    _validate_scalar_decomposition(feedback, effective, values, projections, fallback)
    return {
        "strengths": values,
        "projectors": projections,
        "effective_feedback": effective,
        "symmetric_feedback_fallback": fallback,
        "feedback_approximation_relative_error": float(
            np.linalg.norm(effective - feedback) / max(np.linalg.norm(feedback), 1e-30)
        ),
        "eigenbasis_condition": condition,
    }


def _assemble(covariance, positions, science, weights, gains, h, decomposition, angle):
    """Apply the two projector sums without commuting reconstructor blocks."""
    dtype = weights.dtype
    blocks = weights.reshape(2, len(positions), 2).transpose(1, 0, 2)
    strengths = decomposition["strengths"]
    projected = np.einsum("pab,gbc->gpac", decomposition["projectors"], blocks)
    atm, ho, cross = (np.zeros((2, 2), dtype=dtype) for _ in range(3))
    science_atm, science_ho = (
        np.zeros((len(science), 2, 2), dtype=dtype) for _ in range(2)
    )
    for i, (xi, yi) in enumerate(positions):
        for p, b1 in enumerate(strengths):
            left = projected[i, p]
            args = (science[:, 0], science[:, 1], xi, yi, b1)
            science_atm += (
                gains[i] * covariance.C_t_Tt(*args, zenith_angle=angle).value @ left.T
            )
            science_ho += (
                h[i] * covariance.C_t_Tc(*args, zenith_angle=angle).value @ left.T
            )
            for g, (xg, yg) in enumerate(positions):
                for q, b2 in enumerate(strengths):
                    right = projected[g, q].T
                    args = (xi, yi, xg, yg, b1, b2)
                    atm += (
                        gains[i]
                        * gains[g]
                        * left
                        @ covariance.C_Tt_Tt(*args, zenith_angle=angle).value
                        @ right
                    )
                    ho += (
                        h[i]
                        * h[g]
                        * left
                        @ covariance.C_Tc_Tc(*args, zenith_angle=angle).value
                        @ right
                    )
                    cross += (
                        gains[i]
                        * h[g]
                        * left
                        @ covariance.C_Tt_Tc(*args, zenith_angle=angle).value
                        @ right
                    )
    uncorrected = covariance.C_t_t(
        science[:, 0], science[:, 1], science[:, 0], science[:, 1], zenith_angle=angle
    ).value
    atmospheric = uncorrected + atm - science_atm - science_atm.swapaxes(-1, -2)
    difference = science_ho - cross
    correction = ho - difference - difference.swapaxes(-1, -2)
    return {
        "atmospheric_command": atm,
        "ho_command": ho,
        "atmospheric_ho_command": cross,
        "science_atmospheric_command": science_atm,
        "science_ho_command": science_ho,
        "uncorrected": uncorrected,
        "atmospheric": (atmospheric + atmospheric.swapaxes(-1, -2)) / 2,
        "ho": (correction + correction.swapaxes(-1, -2)) / 2,
    }


@dataclass(frozen=True, kw_only=True, eq=False)
class E2ECalibratedModel:
    """Prepared fitted products for standalone LO displacement prediction.

    Construct from loaded shared products, or use load_e2e_calibrated_model for
    a configured, content-bound model. Construction prepares immutable providers
    once at dtype, including nested state, and rejects controller/photometry
    conflicts. No calibration
    subclass, native source, science PSF or MASTSEL runtime is required.

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
        a, b = self.covariance.controller, self.sensing.controller
        if (
            a.delay != b.delay
            or not np.isclose(a.gain, b.gain, rtol=0, atol=0)
            or a.frame_rate != b.frame_rate
        ):
            raise ValueError("covariance and sensing controller settings differ")
        a, b = self.coma.photometry, self.sensing.response.photometry
        if (
            a.magnitude_band != b.magnitude_band
            or a.photons_per_frame_zeropoint != b.photons_per_frame_zeropoint
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
        """Return immutable runtime state at dtype without altering saved products."""
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
        """
        dtype = self.dtype
        positions = _quantity(ngs_positions, u.arcsec, "ngs_positions", 2, dtype=dtype)
        science = _quantity(
            science_positions, u.arcsec, "science_positions", 2, dtype=dtype
        )
        magnitudes = _quantity(magnitudes, u.mag, "magnitudes", 1, dtype=dtype)
        angle = _angle(zenith_angle, dtype=dtype)
        if (
            positions.shape != (len(magnitudes), 2)
            or not len(positions)
            or len(np.unique(positions.value, axis=0)) != len(positions)
        ):
            raise ValueError("distinct NGS positions must align with magnitudes")
        if science.shape[1:] != (2,) or not len(science):
            raise ValueError("science_positions must be nonempty (S,2)")
        sensor = self.sensing.predict(
            positions=positions, magnitudes=magnitudes, zenith_angle=angle
        )
        h = self.coma.predict(magnitude=magnitudes, zenith_angle=angle)
        metrics = self.ngs_metrics.predict(
            zenith_angle=angle, x=positions[:, 0], y=positions[:, 1]
        )
        photons = self.coma.photometry.photons_per_frame(magnitudes)
        prior = self.reconstructor.measurement_noise(
            photons_per_frame=photons, ee=metrics.ee, fwhm=metrics.fwhm
        )
        weights = build_lo_reconstructor(
            self.covariance,
            self.reconstructor,
            ngs_positions=positions,
            zenith_angle=angle,
            photons_per_frame=photons,
            ee=metrics.ee,
            fwhm=metrics.fwhm,
            dtype=dtype,
        )
        blocks = weights.reshape(2, len(positions), 2).transpose(1, 0, 2)
        feedback = np.einsum("g,gab->ab", sensor.gain.value, blocks)
        self.covariance.controller.transition(feedback)
        decomposition = _scalar_decomposition(feedback)
        self.covariance.controller.transition(decomposition["effective_feedback"])
        components = _assemble(
            self.covariance,
            positions,
            science,
            weights,
            sensor.gain.value,
            h.value,
            decomposition,
            angle,
        )
        noiseless = components["atmospheric"] + components["ho"]
        if np.linalg.eigvalsh(noiseless).min() < -1e-8:
            raise ValueError(
                "calibrated noiseless covariance is not positive semidefinite"
            )
        reconstructed_noise = np.einsum(
            "g,gab,gcb->ac", sensor.variance.value, blocks, blocks
        )
        noise = self.covariance.controller.noise_covariance(
            feedback, reconstructed_noise * u.mas**2
        )
        narrow, broad = self.sensing.blur.scales(len(positions))
        scales = np.array([narrow, broad], dtype=dtype)
        weight = self.sensing.blur.mixture_weight
        mixture = noiseless[:, None] + scales[None, :, None, None] * noise.value
        return E2ELowOrderPrediction._create(
            dtype,
            photons_per_frame=photons,
            metrics=metrics,
            coma_response=h,
            gains=sensor.gain,
            noise_variances=sensor.variance,
            noise_prior=prior,
            reconstructor=weights,
            feedback=feedback,
            **decomposition,
            **{key: value * u.mas**2 for key, value in components.items()},
            noiseless=noiseless * u.mas**2,
            noise=noise,
            total=noiseless * u.mas**2 + noise,
            mixture_weights=np.array([weight, 1 - weight], dtype=dtype),
            noise_scales=scales,
            mixture_covariances=mixture * u.mas**2,
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
