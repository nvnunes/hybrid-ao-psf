"""Native calibration extension boundary, separate from runtime model loading."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator

from astropy import units as u

from ._calibration_config import E2ECalibrationConfiguration
from ._calibration_contracts import (
    CalibrationPhotometry,
    ComaWavefronts,
    ComaWfsMeasurements,
    ComaWfsSampling,
    CovarianceTelemetry,
    FixedGainIntegrator,
    ReconstructorNoiseInputs,
    ReconstructorNoiseMeasurements,
    SensingSpot,
    SensingWfsMeasurements,
    SensingWfsSampling,
)


class E2ECalibration(ABC):
    """Configured native observation hooks for the four common calibrations.

    Construct with the parsed shared configuration. The downstream subclass
    owns native source/configuration validation, selection, setup and execution
    resources. Hooks preserve means and paired sample ordering, return owned
    physical records and propagate native failures after cleaning up resources.
    They do not fit models or publish shared products. The hooks are intended
    for common numerical workflows; runtime prediction does not retain this
    object or require any native implementation.
    """

    def __init__(self, configuration: E2ECalibrationConfiguration):
        if not isinstance(configuration, E2ECalibrationConfiguration):
            raise TypeError("configuration must be E2ECalibrationConfiguration")
        self._configuration = configuration

    @property
    def configuration(self) -> E2ECalibrationConfiguration:
        """Return immutable shared paths/effective controls; outputs may not exist."""
        return self._configuration

    @abstractmethod
    def controller(self, zenith_angle: u.Quantity) -> FixedGainIntegrator:
        """Read fixed gain/delay/cadence for this ZA, without using fitted products."""

    @abstractmethod
    def photometry(self, zenith_angle: u.Quantity) -> CalibrationPhotometry:
        """Read the native magnitude convention and incident-count zero point."""

    @abstractmethod
    def read_covariance_telemetry(
        self, zenith_angles: u.Quantity | None = None
    ) -> Iterator[CovarianceTelemetry]:
        """Yield aligned packed histories per selected ZA; None selects all available.

        Validate native compatibility and timing before yielding. Preserve
        independent realization boundaries and means; do not apply LO filtering.
        Native source identities stay downstream, outside these records.
        """

    @abstractmethod
    def read_coma_wavefronts(
        self, zenith_angle: u.Quantity, samples: int
    ) -> ComaWavefronts:
        """Read independent HO samples with aligned coma and piston/TT removed.

        Native setup and pupil/modal/capture-stage checks belong here. Return
        physical OPDs without native objects or cross-calibration setup files.
        """

    @abstractmethod
    def measure_coma_wfs(
        self,
        zenith_angle: u.Quantity,
        wavefronts: ComaWavefronts,
        sampling: ComaWfsSampling,
    ) -> ComaWfsMeasurements:
        """Measure supplied wavefronts at every magnitude/noise draw, preserving pairing.

        Keep reading separate. The hook owns WFS execution and deterministic
        batch seeds; return uncentred angular tilts and noiseless comparisons.
        """

    @abstractmethod
    def measure_sensing_wfs(
        self, zenith_angle: u.Quantity, spot: SensingSpot, sampling: SensingWfsSampling
    ) -> SensingWfsMeasurements:
        """Return zero and paired signed-shift draws for one supplied fixed spot.

        Preserve image origin, incident flux and displacement order. The native
        sensor owns optical/detector conversion; no LO replay or fitting occurs.
        """

    @abstractmethod
    def measure_reconstructor_noise(
        self, zenith_angle: u.Quantity, inputs: ReconstructorNoiseInputs
    ) -> ReconstructorNoiseMeasurements:
        """Evaluate one ordered batch of native pre-filter noise covariances.

        Construct/access this adapter's native resources independently of coma
        outputs. Keep this noise prior distinct from measured WFS variance.
        """
