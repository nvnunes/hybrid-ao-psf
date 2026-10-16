"""Six pooled covariance functions, rectangular building and portable evaluation."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

import numpy as np
from astropy import units as u
from scipy.interpolate import RegularGridInterpolator

from ._calibration_config import CovarianceCalibrationSettings
from ._calibration_contracts import (
    CovarianceTelemetry,
    FixedGainIntegrator,
    _angle,
    _count,
    _PhysicalRecord,
    _quantity,
)
from ._calibration_numerics import settling_cut
from ._calibration_products import (
    controller_header,
    immutable_metadata,
    json_metadata,
    load_product,
    save_product,
)
from ._covariance_grid import rectangular_tables
from ._covariance_statistics import (
    FUNCTIONS,
    NORMALIZATION,
    FilterSettings,
    HistoryArrays,
    PooledCovariances,
    features,
)
from ._covariance_validation import (
    InterpolationDiagnostics,
    joint_query_check,
    pilot_diagnostics,
    precision_report,
)
from ._io import atomic_output_path

_KIND = "hybrid-ao-psf-covariance-functions"
_MEASUREMENTS = "hybrid-ao-psf-covariance-measurements"
_UNITS = {name: spec[2] for name, spec in FUNCTIONS.items()}


def _readonly(value, dtype=None):
    value = np.ascontiguousarray(value, dtype=dtype)
    return np.frombuffer(value.tobytes(), dtype=value.dtype).reshape(value.shape)


def _window(controller, settings):
    cut, envelope = settling_cut(
        controller.gain,
        controller.delay,
        settings.feedback,
        tolerance=settings.settling_tolerance,
        maximum=settings.maximum_discard,
    )
    return FilterSettings(
        settings.feedback, controller.gain, controller.delay, cut
    ), envelope


@dataclass(frozen=True, kw_only=True, eq=False)
class CovarianceCalibrationMeasurements(_PhysicalRecord):
    """One numerical covariance plane, independent of native source layout.

    Attributes:
        zenith_angle: Viewing angle in [0,90) degrees.
        positions: Cartesian angular directions, (directions,2).
        controller: Fixed gain/delay/frame rate used for every independent reset.
        settings: Feedback nodes, settling and whole-realization bootstrap controls.
        discard: Common discarded frame count for all six functions.
        frames: Input frames per independent realization, before discard.
        realizations: Actual number of equally weighted histories.
        tables: Six physical covariance arrays, in direction1,direction2,
            optional feedback1/feedback2,axis1,axis2 order. Tilt uses angular
            units; primary-coma coefficients use unit-pupil-RMS OPD length.
        diagnostics: Immutable numerical, interpolation and sampling diagnostics.

    The population estimator uses one pooled mean across all retained frames
    and realizations. Arrays and stored unit metadata cannot be mutated.
    """

    zenith_angle: u.Quantity
    positions: u.Quantity
    controller: FixedGainIntegrator
    settings: CovarianceCalibrationSettings
    discard: int
    frames: int
    realizations: int
    tables: Mapping[str, u.Quantity]
    diagnostics: Mapping

    def __post_init__(self):
        angle = _angle(self.zenith_angle)
        positions = _quantity(self.positions, u.arcsec, "positions", 2)
        if positions.shape[1] != 2 or len(np.unique(positions.value, axis=0)) != len(
            positions
        ):
            raise ValueError("distinct positions (directions,2) required")
        if not isinstance(self.controller, FixedGainIntegrator) or not isinstance(
            self.settings, CovarianceCalibrationSettings
        ):
            raise TypeError("typed controller and covariance settings required")
        _count(self.frames, "frames", 3)
        _count(self.realizations, "realizations")
        _count(self.discard, "discard", 0)
        window, _ = _window(self.controller, self.settings)
        if self.discard != window.discard or self.frames - self.discard < 3:
            raise ValueError("covariance plane must use its common settled window")
        if set(self.tables) != set(FUNCTIONS):
            raise ValueError("exactly six covariance functions required")
        tables = {}
        for name, (left, right, unit) in FUNCTIONS.items():
            shape = (
                (len(positions), len(positions))
                + ((len(self.settings.feedback),) if left else ())
                + ((len(self.settings.feedback),) if right else ())
                + (2, 2)
            )
            value = _quantity(self.tables[name], u.Unit(unit), name, len(shape))
            if value.shape != shape:
                raise ValueError(
                    f"incorrect covariance table shape for {name}: expected {shape}"
                )
            tables[name] = value
        for name, value in (
            ("zenith_angle", angle),
            ("positions", positions),
            ("tables", MappingProxyType(tables)),
            ("diagnostics", immutable_metadata(self.diagnostics)),
        ):
            object.__setattr__(self, name, value)

    def __getattribute__(self, name):
        value = super().__getattribute__(name)
        if name == "tables":
            return MappingProxyType({key: item.view() for key, item in value.items()})
        return value


def measure_covariance_calibration(
    telemetry: CovarianceTelemetry,
    controller: FixedGainIntegrator,
    *,
    settings: CovarianceCalibrationSettings | None = None,
) -> CovarianceCalibrationMeasurements:
    """Measure six functions from aligned histories with independent LO resets.

    Validate cadence and the common settling window before processing. Preserve
    equal realization weights, population divisors, all x/y and direction cross
    terms and one pooled retained-sample mean. Report bounded same-realization
    interpolation errors and whole-realization sampling uncertainty separately.
    No native modules, files or instrument metadata are required.
    """
    if not isinstance(telemetry, CovarianceTelemetry) or not isinstance(
        controller, FixedGainIntegrator
    ):
        raise TypeError("typed telemetry and controller required")
    settings = CovarianceCalibrationSettings() if settings is None else settings
    if not isinstance(settings, CovarianceCalibrationSettings):
        raise TypeError("typed covariance settings required")
    if telemetry.frame_rate != controller.frame_rate:
        raise ValueError("telemetry and controller frame rates differ")
    window, _ = _window(controller, settings)
    if telemetry.tilt.shape[1] - window.discard < 3:
        raise ValueError("insufficient frames after replay settling")

    def histories():
        for index in range(len(telemetry.tilt)):
            yield HistoryArrays(
                telemetry.tilt[index].to_value(u.mas).transpose(1, 2, 0),
                telemetry.coma[index].to_value(u.nm).transpose(1, 2, 0),
                telemetry.positions.to_value(u.arcsec),
            )

    # The pilot consumes at most two histories; full histories are not duplicated.
    import itertools

    pilot = pilot_diagnostics(
        list(itertools.islice(histories(), 2)),
        window,
        tolerance=settings.settling_tolerance,
        maximum=settings.maximum_discard,
    )
    accumulator = PooledCovariances()
    grams = []
    means = []
    interpolation = None
    for source in histories():
        if interpolation is None:
            interpolation = InterpolationDiagnostics(source, window)
        values = features(source, window)
        accumulator.add(values)
        interpolation.add(source)
        selection = np.linspace(
            0, len(source.positions) - 1, min(3, len(source.positions)), dtype=int
        )
        b_indices = sorted({0, len(window.strengths) // 2, len(window.strengths) - 1})
        t, ft, fc = (a[selection] for a in values)
        vector = np.concatenate(
            [
                t.reshape(-1, t.shape[-1]),
                ft[:, b_indices].reshape(-1, t.shape[-1]),
                fc[:, b_indices].reshape(-1, t.shape[-1]),
            ]
        )
        means.append(vector.mean(-1))
        vector = vector - vector.mean(-1, keepdims=True)
        grams.append(vector @ vector.T / vector.shape[-1])
    tables = accumulator.tables()
    diagnostics = {
        "pilot": pilot,
        "ph_count": len(telemetry.tilt),
        "native_table_bytes": sum(a.nbytes for a in tables.values()),
        "interpolation": interpolation.report(),
        "precision": precision_report(
            np.array(grams),
            np.array(means),
            direction_count=len(selection),
            strength_count=len(b_indices),
            bootstrap_draws=settings.bootstrap_draws,
            seed=settings.seed,
        ),
    }
    for name in ("C_t_t", "C_Tc_Tc", "C_Tt_Tt"):
        table = tables[name]
        transpose = (
            table.transpose(1, 0, 3, 2, 5, 4)
            if table.ndim == 6
            else table.transpose(1, 0, 3, 2)
        )
        diagnostics[name + "_transpose_max_abs"] = float(np.max(abs(table - transpose)))
        if np.linalg.norm(table - transpose) > 1e-12 * max(
            np.linalg.norm(table), np.finfo(float).tiny
        ):
            raise ValueError(f"covariance transpose identity failed for {name}")
    return CovarianceCalibrationMeasurements(
        zenith_angle=telemetry.zenith_angle,
        positions=telemetry.positions,
        controller=controller,
        settings=settings,
        discard=window.discard,
        frames=telemetry.tilt.shape[1],
        realizations=len(telemetry.tilt),
        tables={
            name: values * u.Unit(FUNCTIONS[name][2]) for name, values in tables.items()
        },
        diagnostics=diagnostics,
    )


class _CovarianceBank:
    """Private canonical-unit evaluator used by numerical diagnostics and queries."""

    def __init__(self, arrays):
        self.arrays = {name: _readonly(value) for name, value in arrays.items()}
        self.x = self.arrays["x_arcsec"]
        self.y = self.arrays["y_arcsec"]
        self.zeniths = self.arrays["zenith_deg"]
        self.strengths = self.arrays["strengths"]
        self._interpolators = {}
        for name, (left, right, _) in FUNCTIONS.items():
            axes = (
                [self.arrays["airmass"], self.x, self.y, self.x, self.y]
                + ([self.strengths] if left else [])
                + ([self.strengths] if right else [])
            )
            self._interpolators[name] = RegularGridInterpolator(
                axes, self.arrays[name], bounds_error=True
            )

    def evaluate(self, name, coordinates, zenith_deg):
        fields = np.broadcast_arrays(
            np.asarray(zenith_deg, float), *[np.asarray(x, float) for x in coordinates]
        )
        if any(not np.isfinite(x).all() for x in fields):
            raise ValueError("covariance query must be finite")
        za, *args = fields
        if np.any(za < self.zeniths[0]) or np.any(za > self.zeniths[-1]):
            raise ValueError("zenith outside calibrated support")
        names = (
            ["x1", "y1", "x2", "y2"]
            + (["b1"] if FUNCTIONS[name][0] else [])
            + (["b2"] if FUNCTIONS[name][1] else [])
        )
        axes = (
            [self.x, self.y, self.x, self.y]
            + ([self.strengths] if FUNCTIONS[name][0] else [])
            + ([self.strengths] if FUNCTIONS[name][1] else [])
        )
        for label, value, axis in zip(names, args, axes, strict=True):
            if np.any(value < axis[0]) or np.any(value > axis[-1]):
                raise ValueError(
                    f"covariance query outside calibrated {label} support [{axis[0]}, {axis[-1]}]"
                )
        points = np.column_stack(
            [(1 / np.cos(np.deg2rad(za))).reshape(-1), *[x.reshape(-1) for x in args]]
        )
        result = self._interpolators[name](points).reshape(*za.shape, 2, 2)
        left, right, _ = FUNCTIONS[name]
        scaling = 1
        if left:
            scaling = args[4]
        if right:
            scaling = scaling * args[5 if left else 4]
        return result / np.asarray(scaling)[..., None, None]

    def __getattr__(self, name):
        # Only private numerical checks use the six canonical-unit call forms.
        if name in FUNCTIONS:
            return lambda *args, zenith_deg: self.evaluate(name, args, zenith_deg)
        raise AttributeError(name)


@dataclass(frozen=True, init=False, eq=False)
class CovarianceFunctions:
    """Immutable six-function bank with physical public inputs and outputs.

    Construction is performed by build_covariance_calibration or the strict
    loader. Queries broadcast Cartesian angular coordinates and dimensionless
    feedback strengths, with a physical zenith angle. Outputs have shape
    (...,2,2) in physical x/y order. Extrapolation is rejected; feedback
    normalization is undone once after rectangular interpolation in airmass.
    """

    _bank: _CovarianceBank
    _controller: FixedGainIntegrator
    _discard: int

    def __init__(self):
        raise TypeError(
            "Use build_covariance_calibration or load_covariance_calibration"
        )

    @classmethod
    def _from_arrays(cls, arrays, controller, discard):
        if set(arrays) != set(FUNCTIONS) | {
            "x_arcsec",
            "y_arcsec",
            "strengths",
            "zenith_deg",
            "airmass",
            "measured_mask",
            "source_mapping",
        }:
            raise ValueError("unexpected covariance interpolation arrays")
        for key in ("x_arcsec", "y_arcsec", "strengths", "zenith_deg", "airmass"):
            axis = arrays[key]
            if (
                axis.ndim != 1
                or not len(axis)
                or not np.isfinite(axis).all()
                or np.any(np.diff(axis) <= 0)
            ):
                raise ValueError(f"invalid covariance axis {key}")
        b = arrays["strengths"]
        za = arrays["zenith_deg"]
        if (
            len(b) < 2
            or np.any(b <= 0)
            or np.any(za < 0)
            or np.any(za >= 90)
            or len(arrays["x_arcsec"]) < 2
            or len(arrays["y_arcsec"]) < 2
        ):
            raise ValueError("invalid covariance spatial/feedback/zenith support")
        if not np.allclose(
            arrays["airmass"], 1 / np.cos(np.deg2rad(za)), rtol=1e-12, atol=0
        ):
            raise ValueError("airmass and zenith axes disagree")
        FilterSettings(tuple(b), controller.gain, controller.delay, discard)
        shape = (len(arrays["x_arcsec"]), len(arrays["y_arcsec"]))
        mapping, mask = arrays["source_mapping"], arrays["measured_mask"]
        if (
            mapping.shape != shape
            or mask.shape != shape
            or mapping.dtype.kind not in "iu"
            or mask.dtype != bool
        ):
            raise ValueError("invalid rectangular source mapping")
        count = int(mask.sum())
        if (
            count < 1
            or not np.array_equal(np.sort(mapping[mask]), np.arange(count))
            or np.any(mapping < 0)
            or np.any(mapping >= count)
        ):
            raise ValueError("source mapping must reference actual measured directions")
        for name, (left, right, _) in FUNCTIONS.items():
            expected = (
                (len(za), *shape, *shape)
                + ((len(b),) if left else ())
                + ((len(b),) if right else ())
                + (2, 2)
            )
            if (
                arrays[name].shape != expected
                or arrays[name].dtype != np.float64
                or not np.isfinite(arrays[name]).all()
            ):
                raise ValueError(f"invalid normalized covariance table {name}")
        result = object.__new__(cls)
        object.__setattr__(result, "_bank", _CovarianceBank(arrays))
        object.__setattr__(result, "_controller", controller)
        object.__setattr__(result, "_discard", discard)
        return result

    @property
    def controller(self) -> FixedGainIntegrator:
        """Return the fixed numerical controller shared by every calibrated plane."""
        return self._controller

    @property
    def discard(self) -> int:
        """Return the common settling cut in saved frames."""
        return self._discard

    @property
    def zenith_angles(self) -> u.Quantity:
        """Return the measured viewing-angle nodes, in degrees."""
        return u.Quantity(self._bank.zeniths, u.deg, copy=False)

    @property
    def x(self) -> u.Quantity:
        """Return the rectangular Cartesian x nodes, in arcseconds."""
        return u.Quantity(self._bank.x, u.arcsec, copy=False)

    @property
    def y(self) -> u.Quantity:
        """Return the rectangular Cartesian y nodes, in arcseconds."""
        return u.Quantity(self._bank.y, u.arcsec, copy=False)

    @property
    def feedback(self) -> np.ndarray:
        """Return immutable increasing scalar feedback nodes."""
        return self._bank.strengths.view()

    @property
    def normalized_tables(self) -> Mapping[str, u.Quantity]:
        """Return immutable tables multiplied by one strength per filtered input.

        Axis order is airmass,x1,y1,x2,y2,optional b1/b2,axis1,axis2.
        Arrays use mas squared, nm squared OPD or mas*nm as appropriate.
        """
        return MappingProxyType(
            {
                name: u.Quantity(self._bank.arrays[name], u.Unit(unit), copy=False)
                for name, (_, _, unit) in FUNCTIONS.items()
            }
        )

    def _evaluate(self, name, coordinates, zenith_angle):
        if not all(
            isinstance(value, u.Quantity) for value in coordinates[:4]
        ) or not isinstance(zenith_angle, u.Quantity):
            raise TypeError(
                "field coordinates and zenith angle must be physical quantities"
            )
        feedback = [
            value.to_value(u.one) if isinstance(value, u.Quantity) else value
            for value in coordinates[4:]
        ]
        args = [value.to_value(u.arcsec) for value in coordinates[:4]] + feedback
        if any(
            np.asarray(value).dtype.kind not in "fiu"
            for value in [*args, zenith_angle.value]
        ):
            raise ValueError("covariance coordinates and feedback must be real")
        return self._bank.evaluate(name, args, zenith_angle.to_value(u.deg)) * u.Unit(
            FUNCTIONS[name][2]
        )

    def C_t_t(
        self,
        x1: u.Quantity,
        y1: u.Quantity,
        x2: u.Quantity,
        y2: u.Quantity,
        *,
        zenith_angle: u.Quantity,
    ) -> u.Quantity:
        """Return unfiltered atmospheric tilt covariance, in mas squared."""
        return self._evaluate("C_t_t", [x1, y1, x2, y2], zenith_angle)

    def C_Tc_Tc(
        self,
        x1: u.Quantity,
        y1: u.Quantity,
        x2: u.Quantity,
        y2: u.Quantity,
        b1,
        b2,
        *,
        zenith_angle: u.Quantity,
    ) -> u.Quantity:
        """Return filtered primary-coma covariance, in nm squared OPD."""
        return self._evaluate("C_Tc_Tc", [x1, y1, x2, y2, b1, b2], zenith_angle)

    def C_t_Tc(
        self,
        x1: u.Quantity,
        y1: u.Quantity,
        x2: u.Quantity,
        y2: u.Quantity,
        b2,
        *,
        zenith_angle: u.Quantity,
    ) -> u.Quantity:
        """Return atmospheric tilt versus filtered coma covariance, in mas*nm."""
        return self._evaluate("C_t_Tc", [x1, y1, x2, y2, b2], zenith_angle)

    def C_Tt_Tc(
        self,
        x1: u.Quantity,
        y1: u.Quantity,
        x2: u.Quantity,
        y2: u.Quantity,
        b1,
        b2,
        *,
        zenith_angle: u.Quantity,
    ) -> u.Quantity:
        """Return filtered tilt versus filtered coma covariance, in mas*nm."""
        return self._evaluate("C_Tt_Tc", [x1, y1, x2, y2, b1, b2], zenith_angle)

    def C_Tt_Tt(
        self,
        x1: u.Quantity,
        y1: u.Quantity,
        x2: u.Quantity,
        y2: u.Quantity,
        b1,
        b2,
        *,
        zenith_angle: u.Quantity,
    ) -> u.Quantity:
        """Return filtered atmospheric tilt covariance, in mas squared."""
        return self._evaluate("C_Tt_Tt", [x1, y1, x2, y2, b1, b2], zenith_angle)

    def C_t_Tt(
        self,
        x1: u.Quantity,
        y1: u.Quantity,
        x2: u.Quantity,
        y2: u.Quantity,
        b2,
        *,
        zenith_angle: u.Quantity,
    ) -> u.Quantity:
        """Return atmospheric tilt versus filtered tilt covariance, in mas squared."""
        return self._evaluate("C_t_Tt", [x1, y1, x2, y2, b2], zenith_angle)


@dataclass(frozen=True, kw_only=True)
class CovarianceCalibrationResult:
    """Standalone covariance build output, without implicit publication.

    Attributes:
        model: Immutable combined multi-zenith six-function interpolator bank.
        diagnostics: Immutable knot/joint-validity and per-plane diagnostics.
    """

    model: CovarianceFunctions
    diagnostics: Mapping

    def __post_init__(self):
        if not isinstance(self.model, CovarianceFunctions):
            raise TypeError("model must be CovarianceFunctions")
        object.__setattr__(self, "diagnostics", immutable_metadata(self.diagnostics))


def build_covariance_calibration(
    planes: Iterable[CovarianceCalibrationMeasurements],
) -> CovarianceCalibrationResult:
    """Combine compatible measured planes into one rectangular covariance bank.

    Require the same directions, cadence, controller, feedback and settled window;
    realization counts may differ. Nearest-node corner filling, normalization
    and interpolation in airmass preserve the accepted numerical procedure.
    Native metadata and sources are neither inspected nor retained.
    """
    planes = sorted(planes, key=lambda plane: plane.zenith_angle.value)
    if not planes or any(
        not isinstance(plane, CovarianceCalibrationMeasurements) for plane in planes
    ):
        raise ValueError("provide accepted covariance measurement planes")
    first = planes[0]
    if len({float(p.zenith_angle.value) for p in planes}) != len(planes):
        raise ValueError("duplicate measured zenith planes")
    rectangular = []
    for plane in planes:
        if (
            not np.array_equal(plane.positions.value, first.positions.value)
            or plane.settings != first.settings
            or plane.controller.gain != first.controller.gain
            or plane.controller.delay != first.controller.delay
            or plane.controller.frame_rate != first.controller.frame_rate
            or plane.frames != first.frames
            or plane.discard != first.discard
        ):
            raise ValueError(
                "incompatible covariance directions, timing, controller or measurement settings"
            )
        rectangular.append(
            rectangular_tables(
                {
                    **{name: value.value for name, value in plane.tables.items()},
                    "positions_arcsec": plane.positions.value,
                    "strengths": np.array(plane.settings.feedback),
                }
            )
        )
    reference = rectangular[0]
    arrays = {
        name: np.stack([item[name] for item in rectangular]) for name in FUNCTIONS
    }
    arrays.update(
        {
            name: reference[name]
            for name in (
                "x_arcsec",
                "y_arcsec",
                "strengths",
                "measured_mask",
                "source_mapping",
            )
        }
    )
    angles = np.array([plane.zenith_angle.value for plane in planes])
    arrays.update(zenith_deg=angles, airmass=1 / np.cos(np.deg2rad(angles)))
    model = CovarianceFunctions._from_arrays(arrays, first.controller, first.discard)
    errors = {}
    for plane in planes:
        points = plane.positions.value
        for name, (left, right, _) in FUNCTIONS.items():
            maximum = 0.0
            for i, j in (
                (0, 0),
                (0, len(points) - 1),
                (len(points) // 2, len(points) // 2),
            ):
                args = [*points[i], *points[j]]
                index = [i, j]
                if left:
                    args.append(plane.settings.feedback[0])
                    index.append(0)
                if right:
                    args.append(plane.settings.feedback[-1])
                    index.append(len(plane.settings.feedback) - 1)
                actual = model._bank.evaluate(name, args, plane.zenith_angle.value)
                expected = plane.tables[name].value[tuple(index)]
                scale = max(float(np.linalg.norm(expected)), np.finfo(float).tiny)
                maximum = max(maximum, float(np.linalg.norm(actual - expected) / scale))
            if maximum > 1e-12:
                raise ValueError(f"interpolation knot recovery failed for {name}")
            errors[f"za{plane.zenith_angle.value:g}_{name}"] = maximum
    return CovarianceCalibrationResult(
        model=model,
        diagnostics={
            "knot_scaled_errors": errors,
            "za_count": len(planes),
            "array_bytes": sum(value.nbytes for value in arrays.values()),
            "joint_query": joint_query_check(model._bank),
            "plane_diagnostics": [json_metadata(plane.diagnostics) for plane in planes],
        },
    )


def save_covariance_calibration(model: CovarianceFunctions, path: str | Path) -> None:
    """Publish one combined runtime bank at model.json, retaining measurements."""
    if not isinstance(model, CovarianceFunctions):
        raise TypeError("model must be CovarianceFunctions")
    save_product(
        path,
        _KIND,
        {
            "units": _UNITS,
            "normalization": NORMALIZATION,
            "controller": controller_header(model.controller),
            "discard": model.discard,
        },
        model._bank.arrays,
    )


def load_covariance_calibration(path: str | Path) -> CovarianceFunctions:
    """Load a strict immutable bank without source histories or E2E software."""
    header, arrays = load_product(path, _KIND)
    if (
        set(header)
        != {
            "schema",
            "version",
            "payload",
            "payload_sha256",
            "units",
            "normalization",
            "controller",
            "discard",
        }
        or header["units"] != _UNITS
        or header["normalization"] != NORMALIZATION
        or set(header["controller"]) != {"gain", "delay", "frame_rate_hz"}
    ):
        raise ValueError("invalid covariance product fields, units or normalization")
    controller = FixedGainIntegrator(
        gain=header["controller"]["gain"],
        delay=header["controller"]["delay"],
        frame_rate=header["controller"]["frame_rate_hz"] * u.Hz,
    )
    _count(header["discard"], "discard", 0)
    return CovarianceFunctions._from_arrays(arrays, controller, header["discard"])


def save_covariance_measurements(
    measurements: CovarianceCalibrationMeasurements, path: str | Path
) -> None:
    """Atomically save one measured plane with numerical settings and diagnostics."""
    if not isinstance(measurements, CovarianceCalibrationMeasurements):
        raise TypeError("unsupported covariance measurement type")
    header = {
        "schema": _MEASUREMENTS,
        "version": 1,
        "units": _UNITS,
        "zenith_angle_deg": float(measurements.zenith_angle.value),
        "controller": controller_header(measurements.controller),
        "settings": measurements.settings.__dict__,
        "discard": measurements.discard,
        "frames": measurements.frames,
        "realizations": measurements.realizations,
        "diagnostics": json_metadata(measurements.diagnostics),
    }
    arrays = {name: value.value for name, value in measurements.tables.items()}
    with atomic_output_path(Path(path)) as temporary, temporary.open("wb") as stream:
        np.savez_compressed(
            stream,
            __header__=json.dumps(header, allow_nan=False),
            positions_arcsec=measurements.positions.value,
            **arrays,
        )


def load_covariance_measurements(path: str | Path) -> CovarianceCalibrationMeasurements:
    """Load strict reusable numerical measurements; never open native sources."""
    with np.load(Path(path), allow_pickle=False) as saved:
        header = json.loads(str(saved["__header__"]))
        expected = {
            "schema",
            "version",
            "units",
            "zenith_angle_deg",
            "controller",
            "settings",
            "discard",
            "frames",
            "realizations",
            "diagnostics",
        }
        if (
            set(header) != expected
            or header["schema"] != _MEASUREMENTS
            or type(header["version"]) is not int
            or header["version"] != 1
            or header["units"] != _UNITS
            or set(saved.files) != set(FUNCTIONS) | {"__header__", "positions_arcsec"}
        ):
            raise ValueError("invalid covariance measurement schema/units/fields")
        controller = header["controller"]
        if set(controller) != {"gain", "delay", "frame_rate_hz"}:
            raise ValueError("invalid covariance controller fields")
        return CovarianceCalibrationMeasurements(
            zenith_angle=header["zenith_angle_deg"] * u.deg,
            positions=saved["positions_arcsec"] * u.arcsec,
            controller=FixedGainIntegrator(
                gain=controller["gain"],
                delay=controller["delay"],
                frame_rate=controller["frame_rate_hz"] * u.Hz,
            ),
            settings=CovarianceCalibrationSettings(**header["settings"]),
            discard=header["discard"],
            frames=header["frames"],
            realizations=header["realizations"],
            tables={
                name: saved[name] * u.Unit(unit)
                for name, (_, _, unit) in FUNCTIONS.items()
            },
            diagnostics=header["diagnostics"],
        )
