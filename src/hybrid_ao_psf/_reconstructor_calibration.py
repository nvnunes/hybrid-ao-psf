"""Reusable pre-filter noise measurements and the calibrated LO noise prior."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from astropy import units as u
from scipy.interpolate import RegularGridInterpolator

from ._calibration_contracts import (
    ReconstructorNoiseInputs,
    ReconstructorNoiseMeasurements,
    _count,
    _PhysicalRecord,
    _quantity,
)
from ._calibration_products import immutable_metadata, load_product, save_product
from ._io import atomic_output_path

_KIND = "hybrid-ao-psf-reconstructor-noise"
_MEASUREMENTS = "hybrid-ao-psf-reconstructor-measurements"


def _axis(value, unit, name):
    value = _quantity(value, unit, name, 1)
    if len(value) < 2 or np.any(value.value <= 0) or np.any(np.diff(value.value) <= 0):
        raise ValueError(f"{name} requires >=2 increasing positive nodes")
    return value


@dataclass(frozen=True, kw_only=True, eq=False)
class ReconstructorCalibrationMeasurements(_PhysicalRecord):
    """Complete ordered grid, holdout and equal-signal validation observations.

    Attributes:
        photons_per_frame: Increasing count axis; P*EE signal when ee is None.
        fwhm: Increasing angular width axis.
        ee: Increasing fractions in (0,1], or None for the product representation.
        inputs: Ordered flattened grid, random holdouts, then 24 triples with
            identical P*EE and width but distinct P and EE.
        covariance: Matching (queries,2,2) angular-squared observations.
        holdouts: Positive number of independent interpolation checks.
        seed: Nonnegative generator seed used for query construction.

    No detector settings or native files are retained. The full validation
    observations are required to justify isotropy and the product reduction.
    """

    photons_per_frame: u.Quantity
    fwhm: u.Quantity
    inputs: ReconstructorNoiseInputs
    covariance: u.Quantity
    holdouts: int
    seed: int
    ee: u.Quantity | None = None

    def __post_init__(self):
        photons = _axis(self.photons_per_frame, u.photon, "photons_per_frame")
        widths = _axis(self.fwhm, u.mas, "fwhm")
        encircled = None if self.ee is None else _axis(self.ee, u.one, "ee")
        if encircled is not None and np.any(encircled.value > 1):
            raise ValueError("ee must be <=1")
        _count(self.holdouts, "holdouts")
        _count(self.seed, "seed", 0)
        if not isinstance(self.inputs, ReconstructorNoiseInputs):
            raise TypeError("inputs must be ReconstructorNoiseInputs")
        covariance = ReconstructorNoiseMeasurements(
            covariance=self.covariance
        ).covariance
        if covariance.shape[0] != len(self.inputs.ee):
            raise ValueError("noise observations must align with ordered inputs")
        object.__setattr__(self, "photons_per_frame", photons)
        object.__setattr__(self, "fwhm", widths)
        object.__setattr__(self, "ee", encircled)
        object.__setattr__(self, "covariance", covariance)
        expected = reconstructor_calibration_inputs(
            photons, widths, ee=encircled, holdouts=self.holdouts, seed=self.seed
        )
        for name in ("photons_per_frame", "ee", "fwhm"):
            if not np.array_equal(
                getattr(expected, name).value, getattr(self.inputs, name).value
            ):
                raise ValueError(
                    "measurement inputs differ from the declared ordered grid/holdouts/reduction queries"
                )


def reconstructor_calibration_inputs(
    photons_per_frame, fwhm, *, ee=None, holdouts=1200, seed=20261014
):
    """Construct the deterministic ordered batch for pre-filter noise calibration.

    Count/width inputs are increasing physical axes. ee=None uses a P*EE axis;
    a supplied increasing fraction axis keeps P and EE separate. The batch
    contains flattened ij-grid nodes, seeded holdouts and 24 equal-signal triples.
    This function requires no calibration object or native software.
    """
    photons = _axis(photons_per_frame, u.photon, "photons_per_frame").value
    widths = _axis(fwhm, u.mas, "fwhm").value
    encircled = None if ee is None else _axis(ee, u.one, "ee").value
    if encircled is not None and np.any(encircled > 1):
        raise ValueError("ee must be <=1")
    _count(holdouts, "holdouts")
    _count(seed, "seed", 0)
    axes = (
        {"signal": photons, "fwhm": widths}
        if ee is None
        else {"photons": photons, "ee": encircled, "fwhm": widths}
    )
    grid = np.meshgrid(*axes.values(), indexing="ij")
    rng = np.random.default_rng(seed)
    random = {
        name: rng.uniform(a[0], a[-1], holdouts)
        if name == "ee"
        else np.exp(rng.uniform(np.log(a[0]), np.log(a[-1]), holdouts))
        for name, a in axes.items()
    }
    values = {
        name: np.r_[part.ravel(), random[name]]
        for name, part in zip(axes, grid, strict=True)
    }
    counts = values["signal"] if ee is None else values["photons"]
    fractions = np.ones_like(counts) if ee is None else values["ee"]
    bounds = photons[[0, -1]] if ee is None else photons[[0, -1]] * encircled[[0, -1]]
    signal = np.repeat(np.geomspace(*bounds, 6), 12)
    product_width = np.tile(np.repeat(np.geomspace(*widths[[0, -1]], 4), 3), 6)
    product_ee = np.tile([0.2, 0.5, 1.0], 24)
    return ReconstructorNoiseInputs(
        photons_per_frame=np.r_[counts, signal / product_ee] * u.photon,
        ee=np.r_[fractions, product_ee] * u.one,
        fwhm=np.r_[values["fwhm"], product_width] * u.mas,
    )


@dataclass(frozen=True, kw_only=True, eq=False)
class ReconstructorNoiseModel(_PhysicalRecord):
    """Immutable rectangular, isotropic pre-filter noise prior.

    Attributes:
        photons_per_frame: Positive increasing count axis, P*EE when ee=None.
        fwhm: Positive increasing angular width axis.
        variance: Positive angular-squared grid with ij axis ordering.
        ee: Optional increasing EE axis between photon count and width.
        interpolation: linear or log-linear; counts, width and variance are
            logarithmic in the latter, while EE remains linear.

    This is the centred prior used to choose reconstruction weights, not the
    measured WFS variance used in noise-command propagation. Queries never
    extrapolate and require no native configuration or controller settings.
    """

    photons_per_frame: u.Quantity
    fwhm: u.Quantity
    variance: u.Quantity
    ee: u.Quantity | None = None
    interpolation: str = "log-linear"
    _interpolator: object = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        photons = _axis(self.photons_per_frame, u.photon, "photons_per_frame")
        widths = _axis(self.fwhm, u.mas, "fwhm")
        ee = None if self.ee is None else _axis(self.ee, u.one, "ee")
        if ee is not None and np.any(ee.value > 1):
            raise ValueError("ee must be <=1")
        axes = (
            (photons.value, widths.value)
            if ee is None
            else (photons.value, ee.value, widths.value)
        )
        variance = _quantity(self.variance, u.mas**2, "variance", len(axes))
        if variance.shape != tuple(map(len, axes)) or np.any(variance.value <= 0):
            raise ValueError("variance must be positive and match the declared grid")
        if self.interpolation not in ("linear", "log-linear"):
            raise ValueError("interpolation must be linear or log-linear")
        logarithmic = self.interpolation == "log-linear"
        coordinates = tuple(
            np.log(axis) if logarithmic and not (len(axes) == 3 and i == 1) else axis
            for i, axis in enumerate(axes)
        )
        interpolator = RegularGridInterpolator(
            coordinates,
            np.log(variance.value) if logarithmic else variance.value,
            bounds_error=True,
        )
        for name, value in (
            ("photons_per_frame", photons),
            ("fwhm", widths),
            ("ee", ee),
            ("variance", variance),
            ("_interpolator", interpolator),
        ):
            object.__setattr__(self, name, value)

    def predict(
        self, *, photons_per_frame: u.Quantity, ee: u.Quantity, fwhm: u.Quantity
    ) -> u.Quantity:
        """Return broadcast (...,2,2) angular covariance blocks in physical x/y.

        Positive counts and widths and EE in (0,1] must lie inside table support.
        Product tables bound P*EE, rather than P and EE separately. No additional
        scientific approximation or extrapolation is applied during evaluation.
        """
        if not all(isinstance(v, u.Quantity) for v in (photons_per_frame, ee, fwhm)):
            raise TypeError("noise queries require physical quantities")
        photons, fraction, width = np.broadcast_arrays(
            photons_per_frame.to_value(u.photon),
            ee.to_value(u.one),
            fwhm.to_value(u.mas),
        )
        if any(
            v.dtype.kind not in "fiu" or not np.isfinite(v).all() or np.any(v <= 0)
            for v in (photons, fraction, width)
        ) or np.any(fraction > 1):
            raise ValueError("positive counts/FWHM and EE in (0,1] required")
        values = (
            (photons * fraction, width)
            if self.ee is None
            else (photons, fraction, width)
        )
        axes = (
            (self.photons_per_frame.value, self.fwhm.value)
            if self.ee is None
            else (self.photons_per_frame.value, self.ee.value, self.fwhm.value)
        )
        names = (
            ("signal", "fwhm")
            if self.ee is None
            else ("photons_per_frame", "ee", "fwhm")
        )
        for name, value, axis in zip(names, values, axes, strict=True):
            if np.any(value < axis[0]) or np.any(value > axis[-1]):
                raise ValueError(
                    f"noise query outside calibrated {name} support [{axis[0]}, {axis[-1]}]"
                )
        logarithmic = self.interpolation == "log-linear"
        coordinates = [
            np.log(v) if logarithmic and not (self.ee is not None and i == 1) else v
            for i, v in enumerate(values)
        ]
        predicted = self._interpolator(
            np.column_stack([v.ravel() for v in coordinates])
        )
        if logarithmic:
            predicted = np.exp(predicted)
        return predicted.reshape(width.shape)[..., None, None] * np.eye(2) * u.mas**2

    def measurement_noise(
        self, *, photons_per_frame: u.Quantity, ee: u.Quantity, fwhm: u.Quantity
    ) -> u.Quantity:
        """Return block-diagonal C_eta for a nonempty one-dimensional NGS list."""
        blocks = self.predict(photons_per_frame=photons_per_frame, ee=ee, fwhm=fwhm)
        if blocks.ndim != 3 or not len(blocks):
            raise ValueError("C_eta requires a nonempty one-dimensional NGS list")
        return np.diag(np.repeat(blocks.value[:, 0, 0], 2)) * u.mas**2


@dataclass(frozen=True, kw_only=True)
class ReconstructorCalibrationResult:
    """Standalone build output; publishing remains an explicit operation.

    Attributes:
        model: Immutable pre-filter noise prior.
        diagnostics: Immutable node, holdout and representation checks.
    """

    model: ReconstructorNoiseModel
    diagnostics: Mapping

    def __post_init__(self):
        if not isinstance(self.model, ReconstructorNoiseModel):
            raise TypeError("model must be ReconstructorNoiseModel")
        object.__setattr__(self, "diagnostics", immutable_metadata(self.diagnostics))


def build_reconstructor_calibration(
    measurements: ReconstructorCalibrationMeasurements, *, interpolation="log-linear"
) -> ReconstructorCalibrationResult:
    """Validate observations and build the calibrated prior without native access.

    Reject anisotropic/cross-axis priors and unjustified P*EE reduction. Preserve
    population/isotropy thresholds, seeded holdout queries and interpolation
    arithmetic. No files are opened or published by this operation.
    """
    if not isinstance(measurements, ReconstructorCalibrationMeasurements):
        raise TypeError("measurements must be ReconstructorCalibrationMeasurements")
    covariance = measurements.covariance.to_value(u.mas**2)
    variance = np.diagonal(covariance, axis1=-2, axis2=-1)
    if np.any(variance <= 0):
        raise ValueError("nonpositive noise variance")
    scale = np.sqrt(variance[:, 0] * variance[:, 1])
    if np.any(abs(covariance[:, 0, 1]) / scale > 1e-8) or np.any(
        abs(covariance[:, 1, 0]) / scale > 1e-8
    ):
        raise ValueError("cross-axis noise cannot be represented by the scalar prior")
    isotropy = float(np.max(abs(variance[:, 0] / variance[:, 1] - 1)))
    if isotropy > 1e-8:
        raise ValueError("anisotropic noise cannot be represented by the scalar prior")
    scalar = variance.mean(axis=1)
    dimensions = (
        (len(measurements.photons_per_frame), len(measurements.fwhm))
        if measurements.ee is None
        else (
            len(measurements.photons_per_frame),
            len(measurements.ee),
            len(measurements.fwhm),
        )
    )
    count = int(np.prod(dimensions))
    holdouts = measurements.holdouts
    if len(scalar) != count + holdouts + 72:
        raise ValueError("unexpected calibration batch length")
    product = scalar[count + holdouts :].reshape(-1, 3)
    reduction = float(np.max(abs(product / product[:, :1] - 1)))
    if measurements.ee is None and reduction > 1e-10:
        raise ValueError("P*EE reduction is unsupported; use separate inputs")
    model = ReconstructorNoiseModel(
        photons_per_frame=measurements.photons_per_frame,
        fwhm=measurements.fwhm,
        ee=measurements.ee,
        variance=scalar[:count].reshape(dimensions) * u.mas**2,
        interpolation=interpolation,
    )
    prediction = model.predict(
        photons_per_frame=measurements.inputs.photons_per_frame[: count + holdouts],
        ee=measurements.inputs.ee[: count + holdouts],
        fwhm=measurements.inputs.fwhm[: count + holdouts],
    ).value[..., 0, 0]
    node_error = float(np.max(abs(prediction[:count] / scalar[:count] - 1)))
    if node_error > 1e-12:
        raise ValueError("interpolator does not recover grid nodes")
    error = prediction[count:] / scalar[count : count + holdouts] - 1
    return ReconstructorCalibrationResult(
        model=model,
        diagnostics={
            "node_max_relative_error": node_error,
            "holdout_count": holdouts,
            "holdout_median_abs_percent": float(np.median(abs(error)) * 100),
            "holdout_p95_abs_percent": float(np.percentile(abs(error), 95) * 100),
            "holdout_max_abs_percent": float(np.max(abs(error)) * 100),
            "isotropy_max_relative_error": isotropy,
            "product_max_relative_error": reduction,
            "product_reduction_verified": reduction <= 1e-10,
            "product_group_count": 24,
            "native_points": len(scalar),
            "interpolation": interpolation,
            "shape": list(dimensions),
        },
    )


def save_reconstructor_calibration(
    model: ReconstructorNoiseModel, path: str | Path
) -> None:
    """Publish the runtime prior at model.json, preserving accepted data on failure."""
    if not isinstance(model, ReconstructorNoiseModel):
        raise TypeError("model must be ReconstructorNoiseModel")
    arrays = {
        "photons_per_frame": model.photons_per_frame.value,
        "fwhm": model.fwhm.value,
        "variance": model.variance.value,
    }
    if model.ee is not None:
        arrays["ee"] = model.ee.value
    save_product(
        path,
        _KIND,
        {
            "representation": "product" if model.ee is None else "separate",
            "interpolation": model.interpolation,
            "units": {
                "photons_per_frame": "ph",
                "fwhm": "mas",
                "variance": "mas2",
                "ee": "",
            },
        },
        arrays,
    )


def load_reconstructor_calibration(path: str | Path) -> ReconstructorNoiseModel:
    """Load a strict shared prior without detector metadata or native software."""
    header, arrays = load_product(path, _KIND)
    if set(header) != {
        "schema",
        "version",
        "payload",
        "payload_sha256",
        "representation",
        "interpolation",
        "units",
    } or header["units"] != {
        "photons_per_frame": "ph",
        "fwhm": "mas",
        "variance": "mas2",
        "ee": "",
    }:
        raise ValueError("invalid reconstructor model fields/units")
    representation = header["representation"]
    if (
        representation not in ("product", "separate")
        or arrays is None
        or set(arrays)
        != {
            "photons_per_frame",
            "fwhm",
            "variance",
            *(["ee"] if representation == "separate" else []),
        }
    ):
        raise ValueError("invalid reconstructor axes/representation")
    return ReconstructorNoiseModel(
        photons_per_frame=arrays["photons_per_frame"] * u.photon,
        fwhm=arrays["fwhm"] * u.mas,
        variance=arrays["variance"] * u.mas**2,
        ee=None if representation == "product" else arrays["ee"] * u.one,
        interpolation=header["interpolation"],
    )


def save_reconstructor_measurements(
    measurements: ReconstructorCalibrationMeasurements, path: str | Path
) -> None:
    """Atomically retain the complete ordered physical batch for offline building."""
    if not isinstance(measurements, ReconstructorCalibrationMeasurements):
        raise TypeError("unsupported measurement type")
    header = {
        "schema": _MEASUREMENTS,
        "version": 1,
        "holdouts": measurements.holdouts,
        "seed": measurements.seed,
        "representation": "product" if measurements.ee is None else "separate",
        "units": {"count": "ph", "fwhm": "mas", "ee": "", "covariance": "mas2"},
    }
    arrays = {
        "photons_per_frame": measurements.photons_per_frame.value,
        "fwhm": measurements.fwhm.value,
        "input_photons": measurements.inputs.photons_per_frame.value,
        "input_ee": measurements.inputs.ee.value,
        "input_fwhm": measurements.inputs.fwhm.value,
        "covariance": measurements.covariance.value,
    }
    if measurements.ee is not None:
        arrays["ee"] = measurements.ee.value
    import json

    with atomic_output_path(Path(path)) as temporary, temporary.open("wb") as stream:
        np.savez_compressed(stream, __header__=json.dumps(header), **arrays)


def load_reconstructor_measurements(
    path: str | Path,
) -> ReconstructorCalibrationMeasurements:
    """Load strict physical measurement arrays; never inspect native acquisition files."""
    import json

    with np.load(Path(path), allow_pickle=False) as saved:
        header = json.loads(str(saved["__header__"]))
        if (
            set(header)
            != {"schema", "version", "holdouts", "seed", "representation", "units"}
            or header["schema"] != _MEASUREMENTS
            or type(header["version"]) is not int
            or header["version"] != 1
            or header["units"]
            != {"count": "ph", "fwhm": "mas", "ee": "", "covariance": "mas2"}
        ):
            raise ValueError("invalid reconstructor measurement schema/units")
        product = header["representation"] == "product"
        if header["representation"] not in ("product", "separate") or set(
            saved.files
        ) != {
            "__header__",
            "photons_per_frame",
            "fwhm",
            "input_photons",
            "input_ee",
            "input_fwhm",
            "covariance",
            *(["ee"] if not product else []),
        }:
            raise ValueError("invalid reconstructor measurement fields")
        return ReconstructorCalibrationMeasurements(
            photons_per_frame=saved["photons_per_frame"] * u.photon,
            fwhm=saved["fwhm"] * u.mas,
            ee=None if product else saved["ee"] * u.one,
            inputs=ReconstructorNoiseInputs(
                photons_per_frame=saved["input_photons"] * u.photon,
                ee=saved["input_ee"] * u.one,
                fwhm=saved["input_fwhm"] * u.mas,
            ),
            covariance=saved["covariance"] * u.mas**2,
            holdouts=header["holdouts"],
            seed=header["seed"],
        )


def solve_lo_reconstructor(
    ngs_covariance: u.Quantity,
    centre_ngs_covariance: u.Quantity,
    measurement_noise: u.Quantity,
) -> u.Quantity:
    """Solve absolute field-centre LO weights from explicit angular covariances.

    NGS/NGS and noise inputs have shape (2*N,2*N); field-centre/NGS has shape
    (2,2*N), in interleaved physical x/y order. Convert all to mas squared and
    solve R=Cas(Css+C_eta)^-1 without conditioning or block normalization. One
    NGS returns I. Return an immutable dimensionless (2,2*N) quantity. No
    calibrated model or native software is required; singular matrices raise.
    """
    from ._calibration_numerics import lo_reconstructor

    values = [
        _quantity(value, u.mas**2, name, 2).value
        for value, name in (
            (ngs_covariance, "ngs_covariance"),
            (centre_ngs_covariance, "centre_ngs_covariance"),
            (measurement_noise, "measurement_noise"),
        )
    ]
    return _quantity(lo_reconstructor(*values) * u.one, u.one, "weights", 2)


def build_lo_reconstructor(
    covariances,
    noise,
    *,
    ngs_positions: u.Quantity,
    zenith_angle: u.Quantity,
    photons_per_frame: u.Quantity,
    ee: u.Quantity,
    fwhm: u.Quantity,
) -> np.ndarray:
    """Return absolute dimensionless field-centre LO weights, shape (2,2*N).

    Use the calibrated unfiltered atmospheric covariance and separate centred
    pre-filter noise prior. Physical inputs are an (N,2) angular direction list,
    scalar zenith angle and per-NGS photon counts, EE and angular FWHM. One NGS
    returns identity without querying either provider. No sensing gain, measured
    noise variance, weight normalization or covariance conditioning is introduced.
    Unsupported queries and unsolvable covariance matrices propagate errors.
    """
    from ._calibration_contracts import _angle
    from ._calibration_numerics import lo_reconstructor

    positions = _quantity(ngs_positions, u.arcsec, "ngs_positions", 2)
    angle = _angle(zenith_angle)
    if positions.shape[1] != 2:
        raise ValueError("ngs_positions must have shape (N,2)")
    if not all(
        isinstance(value, u.Quantity) for value in (photons_per_frame, ee, fwhm)
    ):
        raise TypeError("counts, EE and FWHM must be physical quantities")
    counts, fractions, widths = (
        np.atleast_1d(value.to_value(unit))
        for value, unit in ((photons_per_frame, u.photon), (ee, u.one), (fwhm, u.mas))
    )
    n = len(positions)
    if any(
        value.shape != (n,) or not np.isfinite(value).all() or np.any(value <= 0)
        for value in (counts, fractions, widths)
    ) or np.any(fractions > 1):
        raise ValueError("one positive count/FWHM and EE in (0,1] required per NGS")
    if n == 1:
        return np.eye(2)
    if covariances is None or not isinstance(noise, ReconstructorNoiseModel):
        raise ValueError(
            "multiple NGS require calibrated atmospheric and noise providers"
        )
    x, y = positions.T
    blocks = covariances.C_t_t(
        x[:, None], y[:, None], x[None, :], y[None, :], zenith_angle=angle
    ).to_value(u.mas**2)
    centre = covariances.C_t_t(
        0 * u.arcsec, 0 * u.arcsec, x, y, zenith_angle=angle
    ).to_value(u.mas**2)
    css = blocks.transpose(0, 2, 1, 3).reshape(2 * n, 2 * n)
    cas = centre.transpose(1, 0, 2).reshape(2, 2 * n)
    prior = noise.measurement_noise(
        photons_per_frame=counts * u.photon, ee=fractions * u.one, fwhm=widths * u.mas
    ).to_value(u.mas**2)
    return lo_reconstructor(css, cas, prior)


def validate_covariance_reconstructor(
    covariances,
    noise: ReconstructorNoiseModel,
    telemetry,
) -> Mapping:
    """Check calibrated-weight atmospheric assembly against direct matrix replay.

    Use matching uncentred telemetry for one measured ZA plane and the bank's
    settled window. Choose one/two/three measured directions, build absolute
    weights with the calibrated prior, and use a controlled full-matrix sensing
    response whose feedback eigenvalues are bank nodes. This isolates numerical
    assembly from feedback interpolation and physical gain modelling. Compare
    command, science/command and residual covariances using one pooled mean.
    Raise on incompatible timing/directions or discrepancies above rtol=1e-12,
    atol=1e-10 mas squared. Return immutable per-asterism numerical evidence.
    No native files, WFS model or MASTSEL calculation is used.
    """
    from ._calibration_contracts import CovarianceTelemetry
    from ._covariance_calibration import CovarianceFunctions

    if (
        not isinstance(covariances, CovarianceFunctions)
        or not isinstance(noise, ReconstructorNoiseModel)
        or not isinstance(telemetry, CovarianceTelemetry)
    ):
        raise TypeError("calibrated bank, prior and physical telemetry required")
    if telemetry.frame_rate != covariances.controller.frame_rate:
        raise ValueError("validation and covariance frame rates differ")
    cut = covariances.discard
    if telemetry.tilt.shape[1] - cut < 3:
        raise ValueError("insufficient validation frames after settling")
    nodes = covariances.feedback[[0, -1]]
    feedback = np.diag(nodes)
    chosen = np.linspace(
        0, len(telemetry.positions) - 1, min(3, len(telemetry.positions)), dtype=int
    )
    science_position = telemetry.positions[chosen[0]]
    science = telemetry.tilt.value[:, cut:, chosen[0]]
    width = np.sqrt(noise.fwhm.value[0] * noise.fwhm.value[-1])
    signal = np.sqrt(
        noise.photons_per_frame.value[0] * noise.photons_per_frame.value[-1]
    )
    fraction = (
        0.5 if noise.ee is None else np.sqrt(noise.ee.value[0] * noise.ee.value[-1])
    )
    projections = [np.diag([1, 0]), np.diag([0, 1])]
    angle = telemetry.zenith_angle
    evidence = {}

    def covariance(left, right):
        left, right = left.reshape(-1, 2), right.reshape(-1, 2)
        return (left - left.mean(0)).T @ (right - right.mean(0)) / len(left)

    for count in range(1, len(chosen) + 1):
        indices = chosen[:count]
        positions = telemetry.positions[indices]
        photons = signal / fraction if noise.ee is None else signal
        weights = build_lo_reconstructor(
            covariances,
            noise,
            ngs_positions=positions,
            zenith_angle=angle,
            photons_per_frame=np.full(count, photons) * u.photon,
            ee=np.full(count, fraction) * u.one,
            fwhm=np.full(count, width) * u.mas,
        )
        blocks = [weights[:, 2 * g : 2 * g + 2] for g in range(count)]
        sensing = np.linalg.solve(sum(blocks, np.zeros((2, 2))), feedback)
        transforms = [block @ sensing for block in blocks]
        inputs = sum(
            (
                telemetry.tilt.value[:, :, index] @ transform.T
                for index, transform in zip(indices, transforms, strict=True)
            ),
            np.zeros(telemetry.tilt.shape[:2] + (2,)),
        )
        commands = np.broadcast_to(
            np.linalg.solve(feedback, inputs[:, 0].T).T[:, None], inputs.shape
        ).copy()
        gain, delay = covariances.controller.gain, covariances.controller.delay
        for frame in range(delay, inputs.shape[1]):
            commands[:, frame] = (
                commands[:, frame - 1]
                - gain * (commands[:, frame - delay] @ feedback.T)
                + gain * inputs[:, frame - delay]
            )
        commands = commands[:, cut:]
        q, k = np.zeros((2, 2)), np.zeros((2, 2))
        for (xi, yi), left in zip(positions, transforms, strict=True):
            for p, projection in enumerate(projections):
                a = projection @ left
                k += (
                    covariances.C_t_Tt(
                        *science_position, xi, yi, nodes[p], zenith_angle=angle
                    ).value
                    @ a.T
                )
                for (xg, yg), right in zip(positions, transforms, strict=True):
                    for j, other in enumerate(projections):
                        b = other @ right
                        q += (
                            a
                            @ covariances.C_Tt_Tt(
                                xi, yi, xg, yg, nodes[p], nodes[j], zenith_angle=angle
                            ).value
                            @ b.T
                        )
        residual = (
            covariances.C_t_t(
                *science_position, *science_position, zenith_angle=angle
            ).value
            + q
            - k
            - k.T
        )
        comparisons = {
            "command": (q, covariance(commands, commands)),
            "science_command": (k, covariance(science, commands)),
            "residual": (residual, covariance(science - commands, science - commands)),
        }
        errors = {}
        for name, (predicted, expected) in comparisons.items():
            if not np.allclose(predicted, expected, rtol=1e-12, atol=1e-10):
                raise ValueError(
                    f"calibrated-reconstructor validation failed: {count} NGS, {name}"
                )
            errors[name] = float(np.max(np.abs(predicted - expected)))
        evidence[str(count)] = {
            "weights": weights.tolist(),
            "max_absolute_mas2": errors,
        }
    return immutable_metadata(evidence)
