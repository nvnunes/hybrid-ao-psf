"""Native calibration extension boundary, separate from runtime model loading."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import TYPE_CHECKING

from ao_core.models.control import FixedGainIntegrator
from astropy import units as u

from ._calibration_config import E2ECalibrationConfiguration
from ._calibration_contracts import (
    CalibrationPhotometry,
    ComaWavefronts,
    ComaWfsMeasurements,
    ComaWfsSampling,
    CovarianceTelemetry,
    ReconstructorNoiseInputs,
    ReconstructorNoiseMeasurements,
    SensingSpot,
    SensingWfsMeasurements,
    SensingWfsSampling,
)

if TYPE_CHECKING:
    from ._coma_calibration import ComaCalibrationMeasurements, ComaCalibrationResult
    from ._covariance_calibration import (
        CovarianceCalibrationMeasurements,
        CovarianceCalibrationResult,
    )
    from ._reconstructor_calibration import (
        ReconstructorCalibrationMeasurements,
        ReconstructorCalibrationResult,
    )
    from ._sensing_calibration import SensingCalibrationResult
    from ._sensing_measurements import SensingCalibrationMeasurements
    from ._sensing_noise import SensingNoiseMeasurements, SensingNoiseRequest
    from ._sensing_response import SensingResponseResult


class E2ECalibration(ABC):
    """Configured native calibration-data hooks for the four common calibrations.

    Construct with the parsed shared configuration. The downstream subclass
    owns native source/configuration validation, selection, setup and execution
    resources. Hooks preserve means and paired sample ordering, return owned
    physical records and propagate native failures after cleaning up resources.
    They do not fit models or save shared products. The hooks are intended
    for common numerical workflows; runtime prediction does not retain this
    object or require any native implementation.

    Run reconstructor, covariance, coma and sensing calibration in that
    order when building the full model. Building returns models and diagnostics
    without writing files; save each result explicitly with its save method.
    After collecting the fixed-spot sensing measurements, call ``build_sensing_response()`` with the sampled
    field and magnitude grid before simulating the sensing loop.
    """

    def __init__(self, configuration: E2ECalibrationConfiguration):
        if not isinstance(configuration, E2ECalibrationConfiguration):
            raise TypeError("configuration must be E2ECalibrationConfiguration")
        self._configuration = configuration

    @property
    def configuration(self) -> E2ECalibrationConfiguration:
        """Return immutable shared paths/effective controls; outputs may not exist."""
        return self._configuration

    # Native adapter hooks

    @abstractmethod
    def controller(self, zenith_angle: u.Quantity) -> FixedGainIntegrator:
        """Read the native fixed-gain controller settings.

        Args:
            zenith_angle: Zenith angle that identifies the native setup.

        Returns:
            Gain, frame delay and frame rate from the native setup.
        """

    @abstractmethod
    def photometry(self, zenith_angle: u.Quantity) -> CalibrationPhotometry:
        """Read the native magnitude convention and incident photon-count zero point.

        Args:
            zenith_angle: Zenith angle that identifies the native setup.

        Returns:
            The photometric conversion used to prepare native calibration inputs.
        """

    @abstractmethod
    def measure_reconstructor_noise(
        self, zenith_angle: u.Quantity, inputs: ReconstructorNoiseInputs
    ) -> ReconstructorNoiseMeasurements:
        """Evaluate native pre-filter reconstruction noise at the supplied points.

        The adapter owns native setup and emits covariances in the input row order.

        Args:
            zenith_angle: Zenith angle for the native sensor setup.
            inputs: Ordered calibration rows with photon counts and spot metrics.

        Returns:
            Pre-filter centroid noise covariances. These values are the
            reconstructor prior, distinct from measured sensing variance.
        """

    @abstractmethod
    def read_covariance_telemetry(
        self, zenith_angles: u.Quantity | None = None
    ) -> Iterator[CovarianceTelemetry]:
        """Read aligned, unfiltered atmospheric-tilt and residual-coma histories.

        The adapter validates native source compatibility and timing. Preserve
        the independent realization boundaries and their original means.

        Args:
            zenith_angles: Angular values identifying the requested histories.
                Select all available inputs when omitted.

        Yields:
            One owned physical telemetry record per selected zenith angle.
        """

    @abstractmethod
    def read_coma_wavefronts(
        self, zenith_angle: u.Quantity, samples: int
    ) -> ComaWavefronts:
        """Read independent wavefronts with aligned coma and piston and tip/tilt removed.

        The adapter verifies pupil, modal and capture-stage conventions.

        Args:
            zenith_angle: Zenith angle identifying the native inputs.
            samples: Number of independent wavefront samples to return.

        Returns:
            Aligned physical optical path differences, coma coefficients and pupil.
        """

    @abstractmethod
    def measure_coma_wfs(
        self,
        zenith_angle: u.Quantity,
        wavefronts: ComaWavefronts,
        sampling: ComaWfsSampling,
    ) -> ComaWfsMeasurements:
        """Measure the supplied wavefronts at every magnitude and noise draw.

        Keep the input pairing and use the supplied controls for repeatable draws.

        Args:
            zenith_angle: Zenith angle for the native sensor setup.
            wavefronts: Paired physical wavefronts and coma coefficients.
            sampling: Magnitude grid, draw count and effective seed.

        Returns:
            Uncentred angular tilt draws and paired noiseless comparators.
        """

    @abstractmethod
    def measure_sensing_wfs(
        self, zenith_angle: u.Quantity, spot: SensingSpot, sampling: SensingWfsSampling
    ) -> SensingWfsMeasurements:
        """Measure zero and paired signed-shift draws for one fixed optical spot.

        The adapter converts the source image to native detector units and preserves
        the input coordinates, incident flux and displacement order.

        Args:
            zenith_angle: Zenith angle for the native sensor setup.
            spot: Fixed image and angular displacements with their photometry.
            sampling: Draw counts and effective seed.

        Returns:
            Raw zero-motion and signed-shift WFS measurements with their pairing.
        """

    # Reconstructor calibration

    def measure_reconstructor(
        self,
        zenith_angle: u.Quantity,
        *,
        photons_per_frame: u.Quantity,
        fwhm: u.Quantity,
        ee: u.Quantity | None = None,
    ) -> ReconstructorCalibrationMeasurements:
        """Measure the ordered reconstructor grid and its validation sample.

        Args:
            zenith_angle: Zenith angle at which to prepare the native system.
            photons_per_frame: Increasing photon-count axis. When ``ee`` is omitted,
                the axis represents the product of photon count and encircled energy.
            fwhm: Increasing angular FWHM axis.
            ee: Optional increasing axis of encircled-energy fractions.

        Returns:
            Reusable pre-filter covariance measurements for fitting.

        Raises:
            ValueError: If an input axis or covariance batch is invalid.
        """
        from ._calibration_contracts import _angle
        from ._reconstructor_calibration import (
            ReconstructorCalibrationMeasurements,
            _reconstructor_calibration_inputs,
        )

        zenith_angle = _angle(zenith_angle, dtype=self.configuration.dtype)
        inputs = _reconstructor_calibration_inputs(
            photons_per_frame,
            fwhm,
            ee=ee,
            dtype=self.configuration.dtype,
        )
        measured = self.measure_reconstructor_noise(zenith_angle, inputs)
        return ReconstructorCalibrationMeasurements(
            photons_per_frame=photons_per_frame,
            fwhm=fwhm,
            ee=ee,
            inputs=inputs,
            covariance=measured.covariance,
            dtype=self.configuration.dtype,
        )

    def build_reconstructor(
        self, measurements: ReconstructorCalibrationMeasurements
    ) -> ReconstructorCalibrationResult:
        """Fit the reconstructor noise prior from retained measurements.

        Use the configured interpolation type and the retained measurements.

        Args:
            measurements: An ordered native pre-filter covariance measurement record.

        Returns:
            The fitted prior with its fit and validation diagnostics.
        """
        from ._reconstructor_calibration import build_reconstructor_calibration

        return build_reconstructor_calibration(
            measurements,
            interpolation=self.configuration.reconstructor.interpolation,
            dtype=self.configuration.dtype,
        )

    def save_reconstructor(self, result: ReconstructorCalibrationResult) -> Path:
        """Save the reconstructor noise prior and diagnostics to the configured path.

        Create parent directories as needed. Write the model and diagnostics,
        restoring the previous metadata if saving is interrupted.

        Args:
            result: Model and diagnostics returned by build_reconstructor.

        Returns:
            The absolute configured model.json path.

        Raises:
            TypeError: If result is not a ReconstructorCalibrationResult.
        """
        from ._calibration_workflows import save_result
        from ._reconstructor_calibration import (
            ReconstructorCalibrationResult,
            save_reconstructor_calibration,
        )

        if not isinstance(result, ReconstructorCalibrationResult):
            raise TypeError("result must be ReconstructorCalibrationResult")
        return save_result(
            result,
            self.configuration.reconstructor_model,
            save_reconstructor_calibration,
        )

    # Covariance calibration

    def measure_covariance(
        self, zenith_angles: u.Quantity | None = None
    ) -> Iterator[CovarianceCalibrationMeasurements]:
        """Filter native histories and measure one covariance plane per zenith angle.

        Native source compatibility is checked by the reading hook. Each plane
        uses its native-derived controller and the configured feedback and
        settling controls. Measurements pool a common mean over retained frames
        and realizations.

        Args:
            zenith_angles: Angles to measure. Selects all available inputs when omitted.

        Yields:
            Accepted numerical planes in the order provided by the native reader.
        """
        from ._covariance_calibration import measure_covariance_calibration

        for telemetry in self.read_covariance_telemetry(zenith_angles):
            yield measure_covariance_calibration(
                telemetry,
                self.controller(telemetry.zenith_angle),
                settings=self.configuration.covariance,
                dtype=self.configuration.dtype,
            )

    def build_covariance(
        self,
        measurements: Iterable[CovarianceCalibrationMeasurements],
        *,
        validation_telemetry: Iterable[CovarianceTelemetry] | None = None,
    ) -> CovarianceCalibrationResult:
        """Assemble the covariance interpolators from retained zenith planes.

        Use recorded controller and filtering settings with compatible field
        directions and settled windows.

        Args:
            measurements: Compatible measured planes. Realization counts may differ.
            validation_telemetry: Optional retained histories for direct loop simulation
                with the configured reconstructor prior. Supply one record per
                measured zenith plane.

        Returns:
            The assembled covariance interpolators and its diagnostics.

        Raises:
            FileNotFoundError: If validation needs a missing reconstructor prior.
            ValueError: If measured planes conflict or validation fails to cover them.
        """
        from ._covariance_calibration import build_covariance_calibration

        result = build_covariance_calibration(
            measurements, dtype=self.configuration.dtype
        )
        if validation_telemetry is not None:
            from ._calibration_products import json_metadata
            from ._covariance_calibration import CovarianceCalibrationResult
            from ._reconstructor_calibration import (
                _validate_covariance_reconstructor,
                load_reconstructor_calibration,
            )

            prior = load_reconstructor_calibration(
                self.configuration.reconstructor_model
            )
            checks = {}
            for telemetry in validation_telemetry:
                angle = float(telemetry.zenith_angle.value)
                if angle in checks:
                    raise ValueError("duplicate validation telemetry planes")
                checks[angle] = json_metadata(
                    _validate_covariance_reconstructor(result.model, prior, telemetry)
                )
            if set(checks) != set(result.model.zenith_angles.value):
                raise ValueError(
                    "validation telemetry must cover every measured ZA plane"
                )
            result = CovarianceCalibrationResult(
                model=result.model,
                diagnostics={
                    **json_metadata(result.diagnostics),
                    "calibrated_reconstructor": {
                        str(angle): check for angle, check in checks.items()
                    },
                },
            )
        return result

    def save_covariance(self, result: CovarianceCalibrationResult) -> Path:
        """Save the covariance interpolators and diagnostics to the configured path.

        Create parent directories as needed. Write the model and diagnostics,
        restoring the previous metadata if saving is interrupted.

        Args:
            result: Model and diagnostics returned by build_covariance.

        Returns:
            The absolute configured model.json path.

        Raises:
            TypeError: If result is not a CovarianceCalibrationResult.
        """
        from ._calibration_workflows import save_result
        from ._covariance_calibration import (
            CovarianceCalibrationResult,
            save_covariance_calibration,
        )

        if not isinstance(result, CovarianceCalibrationResult):
            raise TypeError("result must be CovarianceCalibrationResult")
        return save_result(
            result, self.configuration.covariance_model, save_covariance_calibration
        )

    # Coma calibration

    def measure_coma(
        self,
        zenith_angle: u.Quantity,
        magnitudes: u.Quantity,
        *,
        sensing_wavelength: u.Quantity,
    ) -> ComaCalibrationMeasurements:
        """Read paired wavefronts, measure the WFS and retain batch means.

        The configured sample counts and draw controls govern native measurement.

        Args:
            zenith_angle: Zenith angle at which to prepare the native WFS and samples.
            magnitudes: Increasing grid in the adapter's magnitude convention.
            sensing_wavelength: Physical sensing wavelength used for fit conditioning.

        Returns:
            Reusable coma measurements with their photometry and fit controls.

        Raises:
            ValueError: If the sampling grid is invalid or the native WFS returns
                draws on unexpected axes.
        """
        from ._calibration_contracts import (
            ComaResponseMeasurements,
            ComaWfsSampling,
            _angle,
            _quantity,
        )
        from ._coma_calibration import ComaCalibrationMeasurements

        zenith_angle = _angle(zenith_angle, dtype=self.configuration.dtype)
        sensing_wavelength = _quantity(
            sensing_wavelength,
            u.m,
            "sensing_wavelength",
            0,
            dtype=self.configuration.dtype,
        )
        if sensing_wavelength.value <= 0:
            raise ValueError("sensing_wavelength must be positive")
        settings = self.configuration.coma
        sampling = ComaWfsSampling(
            magnitudes=magnitudes,
            noise_draws=settings.noise_draws,
            seed=settings.noise_seed,
            dtype=self.configuration.dtype,
        )
        wavefronts = self.read_coma_wavefronts(zenith_angle, settings.samples)
        measured = self.measure_coma_wfs(zenith_angle, wavefronts, sampling)
        expected = (settings.samples, len(sampling.magnitudes), settings.noise_draws, 2)
        if measured.measured.shape != expected:
            raise ValueError("coma WFS returned different sample/magnitude/draw axes")
        batches = measured.measured.reshape(
            settings.samples, len(sampling.magnitudes), 4, settings.noise_draws // 4, 2
        ).mean(axis=3)
        response = ComaResponseMeasurements(
            coma=wavefronts.coma,
            magnitudes=sampling.magnitudes,
            tilt_mean=batches.mean(axis=2),
            tilt_batches=batches,
            tilt_noiseless=measured.noiseless,
            dtype=self.configuration.dtype,
        )
        return ComaCalibrationMeasurements(
            zenith_angle=zenith_angle,
            response=response,
            sensing_wavelength=sensing_wavelength,
            photometry=self.photometry(zenith_angle),
            settings=settings,
            dtype=self.configuration.dtype,
        )

    def build_coma(
        self, measurements: Iterable[ComaCalibrationMeasurements]
    ) -> ComaCalibrationResult:
        """Fit the joint coma response from retained measurements.

        Verify the fit using the seed recorded with each measurement plane.

        Args:
            measurements: Paired WFS response records at enough zenith angles to
                fit the joint law.

        Returns:
            The fitted response model and its fit diagnostics.

        Raises:
            ValueError: If the measurements have insufficient span or conflict.
        """
        from ._coma_calibration import build_coma_calibration

        return build_coma_calibration(
            measurements,
            dtype=self.configuration.dtype,
        )

    def save_coma(self, result: ComaCalibrationResult) -> Path:
        """Save the joint coma response and diagnostics to the configured path.

        Create parent directories as needed. Write the model and diagnostics,
        restoring the previous metadata if saving is interrupted.

        Args:
            result: Model and diagnostics returned by build_coma.

        Returns:
            The absolute configured model.json path.

        Raises:
            TypeError: If result is not a ComaCalibrationResult.
        """
        from ._calibration_workflows import save_result
        from ._coma_calibration import (
            ComaCalibrationResult,
            save_coma_calibration,
        )

        if not isinstance(result, ComaCalibrationResult):
            raise TypeError("result must be ComaCalibrationResult")
        return save_result(result, self.configuration.coma_model, save_coma_calibration)

    # Sensing calibration

    def _measure_sensing_point(
        self,
        zenith_angle: u.Quantity,
        position: u.Quantity,
        magnitude: u.Quantity,
        spot: SensingSpot,
        *,
        sampling: SensingWfsSampling | None = None,
    ) -> SensingCalibrationMeasurements:
        """Measure one supplied fixed spot at zero and paired signed shifts.

        Args:
            zenith_angle: Zenith angle for the instrument and photometry.
            position: Angular x/y position associated with the spot.
            magnitude: Magnitude in the adapter's convention. Its predicted
                incident photon count must match the spot's count.
            spot: Image fractions per incident photon with physical sampling
                and the requested angular displacements.
            sampling: Optional WFS draw counts and effective point seed.
                Uses the configured controls when omitted.

        Returns:
            A reusable measurement record with physical labels, controller and
            photometry.

        Raises:
            TypeError: If the spot or sampling control has the wrong type.
            ValueError: If the inputs or expected incident photon count disagree.
        """
        import numpy as np

        from ._calibration_contracts import _angle
        from ._sensing_measurements import (
            SensingCalibrationMeasurements,
            _sensing_coordinates,
        )

        if not isinstance(spot, SensingSpot):
            raise TypeError("spot must be SensingSpot")
        zenith_angle = _angle(zenith_angle, dtype=self.configuration.dtype)
        position, magnitude = _sensing_coordinates(position, magnitude)
        if sampling is not None and not isinstance(sampling, SensingWfsSampling):
            raise TypeError("sampling must be SensingWfsSampling")
        settings = self.configuration.sensing
        if sampling is None:
            sampling = SensingWfsSampling(
                zero_draws=settings.zero_draws,
                gain_draws=settings.gain_draws,
                block_draws=settings.block_draws,
                seed=settings.seed,
            )
        photometry = self.photometry(zenith_angle)
        expected = photometry.magnitude_to_photons_per_frame(magnitude)
        if expected.ndim or not np.isclose(
            expected.to_value(u.photon),
            spot.photons_per_frame.to_value(u.photon),
            rtol=1e-12,
            atol=0,
        ):
            raise ValueError(
                "spot photon count differs from declared magnitude/photometry"
            )
        measured = self.measure_sensing_wfs(zenith_angle, spot, sampling)
        return SensingCalibrationMeasurements(
            zenith_angle=zenith_angle,
            position=position,
            magnitude=magnitude,
            samples=measured,
            displacement=spot.displacements[1, 0],
            sampling=sampling,
            controller=self.controller(zenith_angle),
            photometry=photometry,
            dtype=self.configuration.dtype,
        )

    def measure_sensing(
        self,
        zenith_angle: u.Quantity,
        *,
        positions: u.Quantity,
        magnitudes: u.Quantity,
        wavelength: u.Quantity,
        seeds: Iterable[int] | None = None,
    ) -> Iterator[SensingCalibrationMeasurements]:
        """Measure fixed NGS PSFs at the supplied field points and magnitudes.

        The configured PSF file supplies fractions per incident photon in
        physical x/y image orientation. Measurements retain the sampling,
        crop flux and image origin of each interpolated PSF.

        Args:
            zenith_angle: Common zenith angle.
            positions: Angular field positions with shape ``(points, 2)`` in x/y order.
            magnitudes: Magnitudes in the adapter's convention, one per position.
            wavelength: Common sensing wavelength as a scalar length quantity.
            seeds: Optional effective noise seed per input row. Uses the configured
                seed for each row when omitted.

        Yields:
            Raw measurements in input order for independent fitting and noise simulation.

        Raises:
            FileNotFoundError: If the configured NGS image file is missing.
            ValueError: If inputs, interpolation grid or seed count are invalid.
        """
        import numpy as np

        from ._calibration_contracts import _angle, _quantity
        from .interpolation import load_psf_interpolator

        zenith_angle = _angle(zenith_angle, dtype=self.configuration.dtype)
        positions = _quantity(
            positions, u.arcsec, "positions", 2, dtype=self.configuration.dtype
        )
        magnitudes = _quantity(
            magnitudes, u.mag, "magnitudes", 1, dtype=self.configuration.dtype
        )
        wavelength = _quantity(
            wavelength, u.m, "wavelength", 0, dtype=self.configuration.dtype
        )
        if positions.shape != (len(magnitudes), 2) or not len(magnitudes):
            raise ValueError(
                "aligned nonempty (points,2) positions/magnitudes required"
            )
        if wavelength.value <= 0:
            raise ValueError("wavelength must be positive")
        settings = self.configuration.sensing
        seeds = iter([settings.seed] * len(magnitudes) if seeds is None else seeds)
        provider = load_psf_interpolator(
            self.configuration.ngs_psf, dtype=self.configuration.dtype
        )
        photometry = self.photometry(zenith_angle)
        previous = None
        prediction = None
        for position, magnitude in zip(positions, magnitudes, strict=True):
            seed = next(seeds, None)
            if seed is None:
                raise ValueError("one effective seed is required per sensing point")
            if previous is None or not np.array_equal(position.value, previous):
                prediction = provider.evaluate(
                    zenith_angle=zenith_angle,
                    wavelength=wavelength,
                    x=position[0:1],
                    y=position[1:2],
                )
                previous = position.value.copy()
            delta = settings.displacement
            spot = SensingSpot(
                image=prediction.psfs[0],
                pixel_scale=prediction.pixel_scale,
                wavelength=wavelength,
                photons_per_frame=photometry.magnitude_to_photons_per_frame(magnitude),
                displacements=np.array([[0, 0], [1, 0], [-1, 0], [0, 1], [0, -1]])
                * delta,
                dtype=self.configuration.dtype,
            )
            yield self._measure_sensing_point(
                zenith_angle,
                position,
                magnitude,
                spot,
                sampling=SensingWfsSampling(
                    zero_draws=settings.zero_draws,
                    gain_draws=settings.gain_draws,
                    block_draws=settings.block_draws,
                    seed=seed,
                ),
            )
        if next(seeds, None) is not None:
            raise ValueError("too many effective sensing point seeds")

    def simulate_sensing_noise(
        self,
        response: SensingResponseResult,
        requests: Iterable[SensingNoiseRequest],
    ) -> Iterator[SensingNoiseMeasurements]:
        """Simulate noise-driven commands through the configured low-order loop.

        This offline stage loads the configured covariance and reconstructor
        products and yields histories for the later shape fit.

        Args:
            response: Response calibration with retained WFS noise samples.
            requests: Physical asterisms and their spot metrics.

        Yields:
            Complete, reusable command histories in request order.

        Raises:
            FileNotFoundError: If either configured prerequisite product is missing.
            ValueError: If the response, request or prerequisite products conflict.
        """
        from ._covariance_calibration import load_covariance_calibration
        from ._reconstructor_calibration import load_reconstructor_calibration
        from ._sensing_noise import simulate_sensing_noise

        covariance = load_covariance_calibration(self.configuration.covariance_model)
        prior = load_reconstructor_calibration(self.configuration.reconstructor_model)
        yield from simulate_sensing_noise(
            response,
            requests,
            covariance,
            prior,
            settings=self.configuration.sensing,
            dtype=self.configuration.dtype,
        )

    def build_sensing(
        self,
        response: SensingResponseResult,
        noise_measurements: Iterable[SensingNoiseMeasurements],
    ) -> SensingCalibrationResult:
        """Fit the complete sensing and blur model.

        The fit uses the gain and variance tables and weights measurement cases equally
        when fitting the displacement shape.

        Args:
            response: Response calibration that supplies the gain and variance tables.
            noise_measurements: Retained noise histories covering every response zenith plane.

        Returns:
            The fitted sensing model and its fit diagnostics.

        Raises:
            ValueError: If measurement grid is incomplete or inconsistent with the response.
        """
        from ._sensing_calibration import build_sensing_calibration

        return build_sensing_calibration(
            response, noise_measurements, dtype=self.configuration.dtype
        )

    def save_sensing(self, result: SensingCalibrationResult) -> Path:
        """Save the sensing and blur model and diagnostics to the configured path.

        Create parent directories as needed. Write the model and diagnostics,
        restoring the previous metadata if saving is interrupted.

        Args:
            result: Model and diagnostics returned by build_sensing.

        Returns:
            The absolute configured model.json path.

        Raises:
            TypeError: If result is not a SensingCalibrationResult.
        """
        from ._calibration_workflows import save_result
        from ._sensing_calibration import (
            SensingCalibrationResult,
            save_sensing_calibration,
        )

        if not isinstance(result, SensingCalibrationResult):
            raise TypeError("result must be SensingCalibrationResult")
        return save_result(
            result, self.configuration.sensing_model, save_sensing_calibration
        )
