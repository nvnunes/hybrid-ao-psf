"""Native calibration extension boundary, separate from runtime model loading."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator
from typing import TYPE_CHECKING

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
    from ._sensing_replay import SensingReplayMeasurements, SensingReplayRequest
    from ._sensing_response import SensingResponseCalibration


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

    def measure_sensing(
        self,
        zenith_angle: u.Quantity,
        *,
        positions: u.Quantity,
        magnitudes: u.Quantity,
        wavelength: u.Quantity,
        seeds: Iterable[int] | None = None,
    ) -> Iterator[SensingCalibrationMeasurements]:
        """Measure configured NGS HO spots for aligned position/magnitude rows.

        The configured NGS PSF artifact must contain fractions per incident
        photon in physical x/y image orientation. Its preparation/normalization
        belongs downstream. Load it once and preserve cropped flux and sampling;
        never recenter or renormalize images here. Supplied point seeds preserve
        a native recipe's draw schedule; omitted seeds use the configured seed.
        Return one raw observation record at a time, without fitting/replay.
        """
        import numpy as np

        from ._calibration_contracts import _angle, _quantity
        from .interpolation import load_science_ho_psf_interpolator

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
        provider = load_science_ho_psf_interpolator(self.configuration.ngs_psf).prepare(
            dtype=self.configuration.dtype
        )
        photometry = self.photometry(zenith_angle)
        previous = None
        prediction = None
        for position, magnitude in zip(positions, magnitudes, strict=True):
            seed = next(seeds, None)
            if seed is None:
                raise ValueError("one effective seed is required per sensing point")
            if previous is None or not np.array_equal(position.value, previous):
                prediction = provider.predict(
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
                photons_per_frame=photometry.photons_per_frame(magnitude),
                displacements=np.array([[0, 0], [1, 0], [-1, 0], [0, 1], [0, -1]])
                * delta,
                dtype=self.configuration.dtype,
            )
            yield self.measure_sensing_point(
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

    def measure_sensing_point(
        self,
        zenith_angle: u.Quantity,
        position: u.Quantity,
        magnitude: u.Quantity,
        spot: SensingSpot,
        *,
        sampling: SensingWfsSampling | None = None,
    ) -> SensingCalibrationMeasurements:
        """Measure one supplied fixed spot, preserving raw zero/signed WFS draws.

        position/magnitude label the physical input and must match the spot's
        incident photon count under this adapter's photometric convention.
        sampling may supply an explicit effective point seed; omitted controls
        use this configuration. Return reusable observations without fitting,
        replay or publication. Native image normalization remains explicit in
        the supplied SensingSpot light fractions.
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
        expected = photometry.photons_per_frame(magnitude)
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

    def measure_coma(
        self,
        zenith_angle: u.Quantity,
        magnitudes: u.Quantity,
        *,
        sensing_wavelength: u.Quantity,
    ) -> ComaCalibrationMeasurements:
        """Read paired wavefronts, measure the WFS and retain sufficient batch means.

        magnitudes is the increasing grid in this adapter's photometric convention.
        sensing_wavelength is supplied by the native recipe as a physical length;
        it preserves optimizer conditioning, not a phase-based response unit.
        Sample/draw controls come from this configuration. Return reusable
        observations without fitting, publishing or depending on covariance.
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

    def measure_covariance(
        self, zenith_angles: u.Quantity | None = None
    ) -> Iterator[CovarianceCalibrationMeasurements]:
        """Yield one accepted numerical plane at a time from native histories.

        Omitted zenith angles select all available inputs. Native source/cross-ZA
        compatibility belongs to the reading hook. Common filtering, pooled
        measurement and diagnostics use the configured feedback/settling controls
        and each plane's native-derived fixed numerical controller. No model or
        measurement file is published implicitly.
        """
        from ._covariance_calibration import measure_covariance_calibration

        for telemetry in self.read_covariance_telemetry(zenith_angles):
            yield measure_covariance_calibration(
                telemetry,
                self.controller(telemetry.zenith_angle),
                settings=self.configuration.covariance,
                dtype=self.configuration.dtype,
            )

    def measure_reconstructor(
        self,
        zenith_angle: u.Quantity,
        *,
        photons_per_frame: u.Quantity,
        fwhm: u.Quantity,
        ee: u.Quantity | None = None,
    ) -> ReconstructorCalibrationMeasurements:
        """Measure the complete ordered grid/validation batch through the native hook.

        Axes use increasing photon counts, angular FWHM and optional EE fractions.
        ee=None selects the P*EE representation; the count axis then denotes
        signal. Effective holdout count and seed come from this configuration.
        Return reusable physical observations without fitting or publishing.
        """
        from ._calibration_contracts import _angle
        from ._reconstructor_calibration import (
            ReconstructorCalibrationMeasurements,
            reconstructor_calibration_inputs,
        )

        zenith_angle = _angle(zenith_angle, dtype=self.configuration.dtype)
        settings = self.configuration.reconstructor
        inputs = reconstructor_calibration_inputs(
            photons_per_frame,
            fwhm,
            ee=ee,
            holdouts=settings.holdouts,
            seed=settings.seed,
            dtype=self.configuration.dtype,
        )
        measured = self.measure_reconstructor_noise(zenith_angle, inputs)
        return ReconstructorCalibrationMeasurements(
            photons_per_frame=photons_per_frame,
            fwhm=fwhm,
            ee=ee,
            inputs=inputs,
            covariance=measured.covariance,
            holdouts=settings.holdouts,
            seed=settings.seed,
            dtype=self.configuration.dtype,
        )

    def build_reconstructor(
        self, measurements: ReconstructorCalibrationMeasurements
    ) -> ReconstructorCalibrationResult:
        """Build and publish the prior at the configured model.json destination.

        Consume reusable observations without native access. The configured
        interpolation rule controls building; acquisition axes/holdouts remain
        those recorded in measurements. Return the model and diagnostics, and
        save the latter separately beside model.json.
        """
        from ._calibration_workflows import publish_result
        from ._reconstructor_calibration import (
            build_reconstructor_calibration,
            save_reconstructor_calibration,
        )

        result = build_reconstructor_calibration(
            measurements,
            interpolation=self.configuration.reconstructor.interpolation,
            dtype=self.configuration.dtype,
        )
        publish_result(
            result,
            self.configuration.reconstructor_model,
            save_reconstructor_calibration,
        )
        return result

    def build_covariance(
        self,
        measurements: Iterable[CovarianceCalibrationMeasurements],
        *,
        validation_telemetry: Iterable[CovarianceTelemetry] | None = None,
    ) -> CovarianceCalibrationResult:
        """Build and publish one combined covariance bank from retained ZA planes.

        No native sources are opened. Use recorded controller/filter settings,
        permitting different realization counts while requiring compatible
        directions and settled windows. Save diagnostics separately and return
        the built result. Native reuse/source membership remains downstream.
        """
        from ._calibration_workflows import publish_result
        from ._covariance_calibration import (
            build_covariance_calibration,
            save_covariance_calibration,
        )

        result = build_covariance_calibration(
            measurements, dtype=self.configuration.dtype
        )
        if validation_telemetry is not None:
            from ._calibration_products import json_metadata
            from ._covariance_calibration import CovarianceCalibrationResult
            from ._reconstructor_calibration import (
                load_reconstructor_calibration,
                validate_covariance_reconstructor,
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
                    validate_covariance_reconstructor(result.model, prior, telemetry)
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
        publish_result(
            result, self.configuration.covariance_model, save_covariance_calibration
        )
        return result

    def build_coma(
        self, measurements: Iterable[ComaCalibrationMeasurements]
    ) -> ComaCalibrationResult:
        """Build and publish the response law from reusable paired observations.

        Use measured support and recorded fit controls. Require enough ZA planes
        for the joint law; a single newly measured plane does not replace a joint
        model. No wavefronts, WFS resources or other calibrated products are read.
        Save diagnostics separately and return the fitted result.
        """
        from ._calibration_workflows import publish_result
        from ._coma_calibration import build_coma_calibration, save_coma_calibration

        result = build_coma_calibration(
            measurements,
            bootstrap_seed=self.configuration.coma.bootstrap_seed,
            dtype=self.configuration.dtype,
        )
        publish_result(result, self.configuration.coma_model, save_coma_calibration)
        return result

    def replay_sensing(
        self,
        response: SensingResponseCalibration,
        requests: Iterable[SensingReplayRequest],
    ) -> Iterator[SensingReplayMeasurements]:
        """Replay noise using the configured accepted covariance and prior products.

        This is an offline stage: response contains retained WFS noise banks and
        requests contain physical asterisms/metrics. Missing prerequisites raise
        without launching another calibration or accessing native resources.
        Yield complete reusable case histories; publication is explicit.
        """
        from ._covariance_calibration import load_covariance_calibration
        from ._reconstructor_calibration import load_reconstructor_calibration
        from ._sensing_replay import replay_sensing_calibration

        covariance = load_covariance_calibration(self.configuration.covariance_model)
        prior = load_reconstructor_calibration(self.configuration.reconstructor_model)
        yield from replay_sensing_calibration(
            response,
            requests,
            covariance,
            prior,
            settings=self.configuration.sensing,
            dtype=self.configuration.dtype,
        )

    def build_sensing(
        self,
        response: SensingResponseCalibration,
        replays: Iterable[SensingReplayMeasurements],
    ) -> SensingCalibrationResult:
        """Fit and publish the complete sensing/blur product from retained data.

        Gain/variance tables and case-balanced shape fitting use the common
        builders. Require all response ZA planes in the replay inputs. Never
        publish an incomplete response-only runtime model. No native sources or
        preceding calibrations are opened; return model and diagnostics.
        """
        from ._calibration_workflows import publish_result
        from ._sensing_calibration import (
            build_sensing_calibration,
            save_sensing_calibration,
        )

        result = build_sensing_calibration(
            response, replays, dtype=self.configuration.dtype
        )
        publish_result(
            result, self.configuration.sensing_model, save_sensing_calibration
        )
        return result

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
