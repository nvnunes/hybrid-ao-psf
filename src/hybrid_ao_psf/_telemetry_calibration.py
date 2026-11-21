"""Calibrate from files in the Hybrid HDF5 telemetry schema."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, replace
from itertools import combinations, product
from pathlib import Path

import h5py
import numpy as np
from ao_core.io.ini import read_ini, read_ini_array, read_ini_scalar, resolve_ini_path
from ao_core.models.control import FixedGainIntegrator
from astropy import units as u
from numpy.typing import DTypeLike

from ._calibration import _CalibrationWorkflow
from ._calibration_config import (
    _SENSING_ASTERISM_COUNT_KEYS,
    E2ECalibrationConfiguration,
    SensingCalibrationSettings,
    _read_e2e_calibration_configuration,
)
from ._calibration_contracts import (
    CalibrationPhotometry,
    ComaWavefronts,
    ComaWfsMeasurements,
    ComaWfsSampling,
    CovarianceTelemetry,
    ReconstructorNoiseInputs,
    SensingWfsMeasurements,
    SensingWfsSampling,
    _angle,
    _count,
    _quantity,
)
from ._coma_calibration import ComaCalibrationMeasurements, _reduce_coma_measurements
from ._covariance_calibration import (
    CovarianceCalibrationMeasurements,
    measure_covariance_calibration,
)
from ._reconstructor_calibration import (
    ReconstructorCalibrationMeasurements,
)
from ._sensing_measurements import SensingCalibrationMeasurements
from ._sensing_noise import SensingNoiseRequest
from ._sensing_response import (
    SensingResponseResult,
    build_sensing_response,
)

_SIGNED_AXES = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])


@dataclass(frozen=True, kw_only=True, eq=False)
class ZenithTelemetry:
    """The three HDF5 telemetry files captured at one zenith angle.

    Attributes:
        zenith_angle: Scalar observing angle from the zenith.
        covariance_path: Tilt and residual-coma histories in the covariance schema.
        coma_path: Paired wavefronts and WFS draws in the coma schema.
        sensing_path: Fixed-spot WFS draws and spot metrics in the sensing schema.
    """

    zenith_angle: u.Quantity
    covariance_path: Path
    coma_path: Path
    sensing_path: Path

    def __post_init__(self):
        object.__setattr__(
            self, "zenith_angle", _angle(self.zenith_angle, dtype=np.float64)
        )
        for name in ("covariance_path", "coma_path", "sensing_path"):
            object.__setattr__(
                self, name, Path(getattr(self, name)).expanduser().resolve()
            )


class TelemetryCalibration(_CalibrationWorkflow):
    """Read saved HDF5 telemetry and build the four calibration models.

    Read and measure methods return the same reusable records as live calibration.
    Build methods return fitted products and diagnostics. Save each result
    explicitly to the configured destination. Paths identify files in the
    documented HDF5 telemetry schema and are opened by the relevant read step.

    Args:
        configuration: Common calibration controls and model destinations.
        reconstructor_path: HDF5 centroid-noise grid and validation cases.
        planes: Telemetry files with distinct zenith angles, in processing order.
        controller: Fixed integrator used for filtering and noise simulation.
        photometry: NGS magnitude convention and incident photons per WFS frame.
        sensing_wavelength: WFS wavelength as a physical length.
        field_radius: Angular radius of the sensing model's calibrated field.
        obscuration_ratio: Central obscuration diameter divided by pupil diameter.
        asterism_counts: Numbers of one-, two-, and three-NGS asterisms per
            zenith angle. Required for sensing-noise simulation.
    """

    def __init__(
        self,
        configuration: E2ECalibrationConfiguration,
        *,
        reconstructor_path: str | Path,
        planes: Iterable[ZenithTelemetry],
        controller: FixedGainIntegrator,
        photometry: CalibrationPhotometry,
        sensing_wavelength: u.Quantity,
        field_radius: u.Quantity,
        obscuration_ratio: float,
        asterism_counts: tuple[int, int, int] | None = None,
    ):
        super().__init__(configuration)
        dtype = configuration.dtype
        self._reconstructor_path = Path(reconstructor_path).expanduser().resolve()
        self._planes = {}
        for plane in planes:
            if not isinstance(plane, ZenithTelemetry):
                raise TypeError("planes must contain ZenithTelemetry records")
            angle = float(_angle(plane.zenith_angle, dtype=dtype).value)
            if angle in self._planes:
                raise ValueError("Telemetry zenith angles must be distinct.")
            self._planes[angle] = plane

        for family in ("covariance", "coma", "sensing"):
            paths = {
                getattr(plane, f"{family}_path") for plane in self._planes.values()
            }
            if len(paths) != len(self._planes):
                raise ValueError(
                    f"Each zenith angle needs a different {family} telemetry file."
                )

        if not isinstance(controller, FixedGainIntegrator):
            raise TypeError("controller must be a FixedGainIntegrator")
        if not isinstance(photometry, CalibrationPhotometry):
            raise TypeError("photometry must be CalibrationPhotometry")
        self._controller = controller
        self._photometry = photometry
        self._sensing_wavelength = _quantity(
            sensing_wavelength, u.m, "sensing_wavelength", 0, dtype=dtype
        )
        self._field_radius = _quantity(
            field_radius, u.arcsec, "field_radius", 0, dtype=dtype
        )
        if self._sensing_wavelength.value <= 0 or self._field_radius.value <= 0:
            raise ValueError("Sensing wavelength and field radius must be positive.")
        if not 0 <= obscuration_ratio < 1:
            raise ValueError("Obscuration ratio must be in [0, 1).")
        self._obscuration_ratio = obscuration_ratio
        self._asterism_counts = None
        if asterism_counts is not None:
            if len(asterism_counts) != 3:
                raise ValueError(
                    "Supply counts for one-, two-, and three-NGS asterisms."
                )
            for count in asterism_counts:
                _count(count, "asterism count", 0)
            one_ngs, two_ngs, three_ngs = asterism_counts
            if one_ngs == 0 or two_ngs + three_ngs == 0:
                raise ValueError(
                    "The blur fit requires single- and multi-NGS asterisms."
                )
            self._asterism_counts = (one_ngs, two_ngs, three_ngs)

    def read_reconstructor_measurements(self) -> ReconstructorCalibrationMeasurements:
        """Read the ordered grid and check cases, including their measured covariances."""
        with h5py.File(self._reconstructor_path, "r") as telemetry:
            grid = telemetry["grid"]
            checks = telemetry["checks"]
            if "signal" in grid:
                if "flux" in grid or "ee" in grid:
                    raise ValueError(
                        "Use either a signal axis or separate flux and EE axes."
                    )
                signal_axis, ee_axis = grid["signal"][:], None
            else:
                signal_axis, ee_axis = grid["flux"][:], grid["ee"][:]
            fwhm_axis = grid["fwhm"][:]

            check_flux = checks["flux"][:]
            check_ee = checks["ee"][:]
            check_fwhm = checks["fwhm"][:]
            grid_dtypes = [signal_axis.dtype, fwhm_axis.dtype]
            if ee_axis is not None:
                grid_dtypes.append(ee_axis.dtype)
            grid_precision = (
                np.float32
                if any(
                    dtype.kind == "f" and dtype.itemsize <= 4 for dtype in grid_dtypes
                )
                else np.float64
            )
            grid_covariance = grid["covariance"][:]
            check_covariance = checks["covariance"][:]
            dtype = np.result_type(
                np.float32,
                *grid_dtypes,
                check_flux.dtype,
                check_ee.dtype,
                check_fwhm.dtype,
                grid_covariance.dtype,
                check_covariance.dtype,
            )

            # Flatten the grid in the order used by Hybrid's measurement records.
            if ee_axis is None:
                flux, fwhm = np.meshgrid(signal_axis, fwhm_axis, indexing="ij")
                ee = np.ones_like(flux)
            else:
                flux, ee, fwhm = np.meshgrid(
                    signal_axis, ee_axis, fwhm_axis, indexing="ij"
                )
            if grid_covariance.shape != (*flux.shape, 2, 2):
                raise ValueError(
                    "Reconstructor grid covariance does not match its axes."
                )
            inputs = ReconstructorNoiseInputs(
                photons_per_frame=np.r_[flux.ravel(), check_flux] * u.photon,
                ee=np.r_[ee.ravel(), check_ee] * u.one,
                fwhm=np.r_[fwhm.ravel(), check_fwhm] * u.mas,
                dtype=dtype,
            )
            covariance = (
                np.concatenate((grid_covariance.reshape(-1, 2, 2), check_covariance))
                * u.mas**2
            )

        return ReconstructorCalibrationMeasurements(
            photons_per_frame=signal_axis * u.photon,
            fwhm=fwhm_axis * u.mas,
            ee=None if ee_axis is None else ee_axis * u.one,
            inputs=inputs,
            covariance=covariance,
            seed=None,
            grid_precision=grid_precision,
            dtype=dtype,
        )

    def measure_covariance(self) -> Iterator[CovarianceCalibrationMeasurements]:
        """Yield filtered covariance measurements for the configured zenith angles.

        Apply the fixed controller, configured feedback grid and settling checks
        to the aligned tilt and coma histories. Preserve realization boundaries
        and pool one mean over the retained frames.
        """
        for telemetry in self.read_covariance_telemetry():
            yield measure_covariance_calibration(
                telemetry,
                self._controller,
                settings=self.configuration.covariance,
                dtype=self.configuration.dtype,
            )

    def measure_coma(self) -> Iterator[ComaCalibrationMeasurements]:
        """Read and reduce each zenith angle's paired wavefronts and WFS draws.

        Infer the wavefront and draw counts from each file, construct the
        annular pupil on the saved wavefront grid, and retain four detector-batch
        means with the noiseless references for fitting.
        """
        dtype = self.configuration.dtype
        for plane in self._planes.values():
            with h5py.File(plane.coma_path, "r") as telemetry:
                wavefront = telemetry["wavefront"][:]
                coma = telemetry["coma"][:]
                magnitudes = telemetry["magnitudes"][:] * u.mag
                measured = ComaWfsMeasurements(
                    measured=telemetry["tilt"][:] * u.mas,
                    noiseless=telemetry["tilt_noiseless"][:] * u.mas,
                    dtype=dtype,
                )
            if wavefront.ndim != 3 or wavefront.shape[1] != wavefront.shape[2]:
                raise ValueError("Coma telemetry requires square wavefront arrays.")

            coords = np.linspace(-1, 1, wavefront.shape[-1])
            radius_squared = coords[:, None] ** 2 + coords**2
            pupil = (radius_squared <= 1) & (
                radius_squared >= self._obscuration_ratio**2
            )
            wavefronts = ComaWavefronts(
                wavefront=wavefront * u.nm, coma=coma * u.nm, pupil=pupil, dtype=dtype
            )
            settings = replace(
                self.configuration.coma,
                samples=len(wavefronts.coma),
                noise_draws=measured.measured.shape[2],
            )
            sampling = ComaWfsSampling(
                magnitudes=magnitudes,
                noise_draws=settings.noise_draws,
                seed=settings.noise_seed,
                dtype=dtype,
            )
            yield _reduce_coma_measurements(
                _angle(plane.zenith_angle, dtype=dtype),
                sensing_wavelength=self._sensing_wavelength,
                photometry=self._photometry,
                wavefronts=wavefronts,
                measured=measured,
                sampling=sampling,
                settings=settings,
            )

    def read_sensing(self) -> Iterator[SensingCalibrationMeasurements]:
        """Yield fixed-spot WFS draws and spot metrics at each measured grid point.

        Infer the two draw counts and the single complete set of signed shifts
        from each file. Preserve draw pairing while ordering shifts +x, -x,
        +y, -y for gain estimation.
        """
        for plane in self._planes.values():
            yield from self._read_sensing_measurements(plane)

    def build_sensing_response(
        self, measurements: Iterable[SensingCalibrationMeasurements]
    ) -> SensingResponseResult:
        """Fit gain and variance and retain the noise draws and spot metrics.

        The measurement records supply the common photometry, controller,
        and sampling. The configured field radius sets the model's support.
        """
        return build_sensing_response(
            measurements,
            field_radius=self._field_radius,
            dtype=self.configuration.dtype,
        )

    def sensing_noise_requests(
        self, response: SensingResponseResult
    ) -> Iterator[SensingNoiseRequest]:
        """Select the configured number of blur-fit cases from each measured zenith grid.

        Prefer centre and edge stars, close and wide pairs, and spread-out triples,
        with bright, middle and faint magnitudes where measured. Use the EE and
        FWHM captured with the sensing response. Each request has a deterministic
        simulation seed derived from the configured sensing seed.
        """
        if not isinstance(response, SensingResponseResult):
            raise TypeError("response must be a SensingResponseResult")
        if self._asterism_counts is None:
            raise ValueError("Configure asterism counts to select sensing-noise cases.")

        configuration = self.configuration
        next_seed = configuration.sensing.seed
        for zenith_angle in response.model.zenith_angles:
            angle = float(zenith_angle.to_value(u.deg))
            rows = [point[1:] for point in response._noise_samples if point[0] == angle]
            metrics = {
                point[1:]: value
                for point, value in response._spot_metrics.items()
                if point[0] == angle
            }
            if not all(point[:2] in metrics for point in rows):
                raise ValueError(
                    f"Each sensing direction needs spot metrics at ZA {angle:g}."
                )
            for key, positions, magnitudes in _noise_asterisms(
                rows, response.model.field_radius, self._asterism_counts
            ):
                ee, fwhm = np.asarray([metrics[tuple(xy)] for xy in positions.value]).T
                yield SensingNoiseRequest(
                    key=key,
                    population="telemetry",
                    zenith_angle=zenith_angle,
                    positions=positions,
                    magnitudes=magnitudes,
                    ee=ee * u.one,
                    fwhm=fwhm * u.mas,
                    seed=next_seed,
                    dtype=configuration.dtype,
                )
                next_seed += 1

    def read_covariance_telemetry(self) -> Iterator[CovarianceTelemetry]:
        """Read one aligned, uncentred tilt and coma history per configured zenith."""
        for plane in self._planes.values():
            with h5py.File(plane.covariance_path, "r") as telemetry:
                histories = CovarianceTelemetry(
                    zenith_angle=plane.zenith_angle,
                    frame_rate=self._controller.frame_rate,
                    positions=telemetry["positions"][:] * u.arcsec,
                    tilt=telemetry["tilt"][:] * u.mas,
                    coma=telemetry["coma"][:] * u.nm,
                    dtype=self.configuration.dtype,
                )
            yield histories

    def _sensing_simulation_settings(
        self, response: SensingResponseResult
    ) -> SensingCalibrationSettings:
        """Use captured draw counts and displacement with the configured simulation lengths."""
        if not isinstance(response, SensingResponseResult):
            raise TypeError("response must be a SensingResponseResult")
        return replace(
            self.configuration.sensing,
            zero_draws=response._sampling.zero_draws,
            gain_draws=response._sampling.gain_draws,
            block_draws=response._sampling.block_draws,
            displacement=response._displacement,
        )

    def _read_sensing_measurements(self, plane: ZenithTelemetry):
        """Yield one calibration measurement per position and magnitude in the file.

        Keep the draws paired across shifts. The gain estimate uses differences
        between opposite shifts, and the unshifted draws supply the noise estimate.
        """
        settings = self.configuration.sensing
        dtype = self.configuration.dtype
        with h5py.File(plane.sensing_path, "r") as telemetry:
            positions = telemetry["positions"][:] * u.arcsec
            magnitudes = telemetry["magnitudes"][:] * u.mag
            ee = telemetry["ee"][:]
            fwhm = telemetry["fwhm"][:]
            if ee.shape != (len(positions),) or fwhm.shape != (len(positions),):
                raise ValueError(
                    "Sensing telemetry needs one EE and FWHM per direction."
                )

            shifts = telemetry["shifts"][:]
            unshifted = telemetry["tilt_unshifted"]
            shifted = telemetry["tilt_shifted"]
            grid_shape = (len(positions), len(magnitudes))
            displacement = _sensing_displacement(shifts)
            if (
                unshifted.ndim != 4
                or shifted.ndim != 5
                or unshifted.shape[:2] != grid_shape
                or shifted.shape[:2] != grid_shape
                or shifted.shape[2] != len(shifts)
                or unshifted.shape[-1] != 2
                or shifted.shape[-1] != 2
            ):
                raise ValueError(
                    "Sensing tilt arrays do not match the field and shift grids."
                )

            matches = _signed_shift_matches(shifts, displacement.to_value(u.mas))
            shift_rows = np.argmax(matches, axis=1)

            sampling = SensingWfsSampling(
                zero_draws=unshifted.shape[2],
                gain_draws=shifted.shape[3],
                block_draws=settings.block_draws,
            )
            for direction, magnitude in np.ndindex(grid_shape):
                samples = SensingWfsMeasurements(
                    zero=unshifted[direction, magnitude] * u.mas,
                    shifted=shifted[direction, magnitude][shift_rows] * u.mas,
                    dtype=dtype,
                )
                yield SensingCalibrationMeasurements(
                    zenith_angle=plane.zenith_angle,
                    position=positions[direction],
                    magnitude=magnitudes[magnitude],
                    samples=samples,
                    ee=ee[direction] * u.one,
                    fwhm=fwhm[direction] * u.mas,
                    displacement=displacement,
                    sampling=sampling,
                    controller=self._controller,
                    photometry=self._photometry,
                    dtype=dtype,
                )


def _signed_shift_matches(shifts, amplitude):
    """Match the four signed axes to the saved shift rows."""
    offsets = amplitude * _SIGNED_AXES
    return np.isclose(offsets[:, None, :], shifts[None, :, :], rtol=1e-7, atol=0).all(
        axis=2
    )


def _sensing_displacement(shifts):
    """Read the unique amplitude with all four signed shifts present once."""
    if shifts.ndim != 2 or shifts.shape[1] != 2:
        raise ValueError("Sensing telemetry shifts must have shape (shift, 2).")
    positive_x = shifts[(shifts[:, 0] > 0) & (shifts[:, 1] == 0), 0]
    amplitudes = [
        value
        for value in np.unique(positive_x)
        if np.all(np.count_nonzero(_signed_shift_matches(shifts, value), axis=1) == 1)
    ]
    if len(amplitudes) != 1:
        raise ValueError(
            "Sensing telemetry must contain one complete set of signed shifts."
        )
    return amplitudes[0] * u.mas


def _select_grid_cases(preferred, count, ngs_count, position_count, magnitude_count):
    """Choose cases from the preferred set, then other distinct grid combinations."""
    if count <= len(preferred):
        indices = np.linspace(0, len(preferred) - 1, count, dtype=int)
        return [preferred[index] for index in indices]

    selected = list(preferred)
    used = {tuple(sorted(zip(*case))) for case in preferred}
    for brightness in product(range(magnitude_count), repeat=ngs_count):
        for geometry in combinations(range(position_count), ngs_count):
            if tuple(zip(geometry, brightness)) not in used:
                selected.append((geometry, brightness))
            if len(selected) == count:
                return selected
    raise ValueError(f"The sensing grid has fewer than {count} {ngs_count}-NGS cases.")


def _noise_asterisms(rows, field_radius, counts):
    """Choose blur-fit asterisms from the measured directions and magnitudes."""
    values = np.asarray(rows, dtype=field_radius.dtype)
    measured_positions = np.unique(values[:, :2], axis=0)
    radii = np.linalg.norm(measured_positions, axis=1)
    in_field = radii <= field_radius.to_value(u.arcsec)
    positions, radii = measured_positions[in_field], radii[in_field]
    magnitudes = np.unique(values[:, 2])
    largest_asterism = max(
        index for index, count in enumerate(counts, start=1) if count
    )
    if len(positions) < largest_asterism:
        raise ValueError(
            f"The blur fit needs {largest_asterism} measured in-field positions."
        )
    # An HDF5 zenith plane has a common magnitude grid across all of its directions.
    if len(rows) != len(measured_positions) * len(magnitudes):
        raise ValueError(
            "Blur-fit cases need the same measured magnitudes at each direction."
        )

    centre = np.argmin(radii)
    edge = np.argmax(np.where(np.arange(len(radii)) == centre, -np.inf, radii))
    separations = np.linalg.norm(positions[:, None] - positions[None, :], axis=2)
    pairs = np.column_stack(np.triu_indices(len(positions), k=1))
    pair_separations = separations[pairs[:, 0], pairs[:, 1]]
    closest = pairs[np.argmin(pair_separations)]
    widest = pairs[np.argmax(pair_separations)]
    # Start with the widest pair and maximize the distance to its nearer star.
    third = np.argmax(separations[:, widest].min(axis=1))
    triple = (*widest, third)

    bright, middle, faint = 0, len(magnitudes) // 2, len(magnitudes) - 1
    positions = positions * u.arcsec
    magnitudes = magnitudes * u.mag
    preferred_cases = (
        (
            ((centre,), (bright,)),
            ((centre,), (faint,)),
            ((edge,), (bright,)),
            ((edge,), (faint,)),
        ),
        (
            (tuple(closest), (faint, faint)),
            (tuple(widest), (bright, faint)),
        ),
        (
            (triple, (middle, middle, middle)),
            (triple, (bright, middle, faint)),
        ),
    )
    for ngs_count, (count, preferred) in enumerate(
        zip(counts, preferred_cases, strict=True), start=1
    ):
        if not count:
            continue
        preferred = tuple(dict.fromkeys(preferred))
        cases = _select_grid_cases(
            preferred, count, ngs_count, len(positions), len(magnitudes)
        )
        for index, (geometry, brightness) in enumerate(cases, start=1):
            yield (
                f"{ngs_count}-ngs-{index}",
                positions[list(geometry)],
                magnitudes[list(brightness)],
            )


def _telemetry_path(
    inputs: Mapping[str, str], key: str, source: Path, angle: u.Quantity | None = None
) -> Path:
    """Resolve a file path and substitute its angle in degrees when supplied."""
    value = inputs.get(key, "")
    if angle is not None:
        value = value.replace(
            "{ZA}", np.format_float_positional(angle.to_value(u.deg), trim="-")
        )
    return resolve_ini_path(
        value, source=source, label=f"inputs.{key}", must_exist=False
    )


def load_telemetry_calibration(
    path: str | Path, *, dtype: DTypeLike | None = None
) -> TelemetryCalibration:
    """Read an INI for calibration from Hybrid HDF5 telemetry.

    Resolve input and output paths relative to the INI, expanding {ZA} with the
    angle in degrees. Read the physical system settings and the counts of one-,
    two-, and three-NGS asterisms for the noise-blur fit. Telemetry files are
    opened by the calibration methods.

    Args:
        path: INI with the telemetry paths, system settings, and model destinations.
        dtype: Optional float32 or float64 override of the configured precision.

    Returns:
        Calibration ready to read the configured telemetry.

    Raises:
        FileNotFoundError: If the INI is missing.
        ValueError: If the configured paths or settings are missing or invalid.
    """
    source, sections = read_ini(path)
    config = _read_e2e_calibration_configuration(sections, source)
    if dtype is not None:
        config = config.with_overrides(dtype=dtype)
    if any(section not in sections for section in ("inputs", "system")):
        raise ValueError("Telemetry calibration requires [inputs] and [system].")

    inputs = sections["inputs"]
    system = sections["system"]
    angles = read_ini_array(system, "ZenithAngle", label="system.ZenithAngle") * u.deg
    planes = [
        ZenithTelemetry(
            zenith_angle=angle,
            covariance_path=_telemetry_path(
                inputs, "CovarianceTelemetryPath", source, angle
            ),
            coma_path=_telemetry_path(inputs, "ComaTelemetryPath", source, angle),
            sensing_path=_telemetry_path(inputs, "SensingTelemetryPath", source, angle),
        )
        for angle in angles
    ]

    wavelengths = read_ini_array(system, "Wavelength", label="system.Wavelength")
    if wavelengths.size != 1:
        raise ValueError("[system] Wavelength requires one sensing wavelength.")
    delay = read_ini_scalar(
        system, "LoopDelaySteps_LO", label="system.LoopDelaySteps_LO"
    )
    if not delay.is_integer():
        raise ValueError("[system] LoopDelaySteps_LO must be a whole number of frames.")
    controller = FixedGainIntegrator(
        gain=read_ini_scalar(system, "LoopGain_LO", label="system.LoopGain_LO"),
        delay=int(delay),
        frame_rate=read_ini_scalar(
            system, "SensorFrameRate_LO", label="system.SensorFrameRate_LO"
        )
        * u.Hz,
        dtype=config.dtype,
    )
    photometry = CalibrationPhotometry(
        magnitude_band=system.get("NGSMagStandard", ""),
        photons_per_frame_zeropoint=read_ini_scalar(
            system, "NGSMagZeropoint", label="system.NGSMagZeropoint"
        )
        * u.photon,
        dtype=config.dtype,
    )
    counts = None
    if "sensing" in sections:
        sensing = sections["sensing"]
        keys = _SENSING_ASTERISM_COUNT_KEYS
        if any(key in sensing for key in keys):
            if not all(key in sensing for key in keys):
                raise ValueError(
                    "[sensing] requires all three NGS asterism counts when any is set."
                )
            counts = tuple(sensing.getint(key) for key in keys)

    return TelemetryCalibration(
        config,
        reconstructor_path=_telemetry_path(
            inputs, "ReconstructorTelemetryPath", source
        ),
        planes=planes,
        controller=controller,
        photometry=photometry,
        sensing_wavelength=wavelengths[0] * u.m,
        field_radius=read_ini_scalar(system, "FieldRadius", label="system.FieldRadius")
        * u.arcsec,
        obscuration_ratio=read_ini_scalar(
            system, "ObscurationRatio", label="system.ObscurationRatio"
        ),
        asterism_counts=counts,
    )
