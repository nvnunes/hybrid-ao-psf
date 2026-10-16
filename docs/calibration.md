# E2E Calibration Inputs

This API defines the native observation boundary for covariance, coma, sensing
and reconstructor calibration. It provides immutable physical records, shared
configuration, reusable observation persistence and fitted-product content
identities. Native source selection, WFS execution and setup validation belong
to the downstream subclass of `E2ECalibration`.

## Native Hooks

Subclass `E2ECalibration` and implement its controller/photometry accessors and
five reading/measurement hooks. Covariance reads aligned tilt/coma histories;
coma separates wavefront reading from WFS measurement; sensing measures supplied
spots; reconstructor measures pre-filter centroid-noise covariances. Hooks own
native resources and propagate failures. They preserve means and paired sample
order, and perform no shared LO filtering, response estimation or model fitting.
This observation API does not execute a complete calibration pipeline or load
a fitted scientific prediction model.

All physical values use Astropy quantities. Covariance histories have axes
`(realizations, frames, directions, 2)` with angular tilt and primary-coma OPD
length. Coordinates and the last axis are physical x/y. Coma is unit pupil RMS,
orthogonal to piston/tilt, after common HO correction and before science-specific
correction. The native reader verifies those conventions and alignment. Each
history remains independent and retains its mean. Record arrays are copied to
immutable backing; returned quantity views cannot mutate stored unit metadata.

## Shared Configuration

Use `load_e2e_calibration_configuration()` for the shared part of one INI:

```ini
[e2e]
covariance_model = products/covariance/model.json
coma_model = products/coma/model.json
sensing_model = products/sensing/model.json
reconstructor_model = products/reconstructor/model.json
science_psf = inputs/science.pkl
ngs_psf = inputs/ngs-psf.pkl
ngs_metrics = inputs/ngs-metrics.pkl

[calibration.covariance]
feedback = [0.2, 0.5, 1.0, 1.5, 2.0]
bootstrap_draws = 200

[calibration.sensing]
displacement = 10 mas
```

Paths are relative to the INI. Parsing opens neither models nor native sources;
planned output paths need not exist. The loader validates keys in `[e2e]` and
`[calibration.<family>]`; downstream implementations validate their own sections.
Fixed controller settings and the native-derived photon-count zero point come
from the configured adapter, not shared instrument defaults. `CalibrationPhotometry`
converts magnitudes using the supplied convention and incident count zero point,
before detector cropping.

Controls resolve from built-in defaults, then INI values, then explicit values
passed to `configuration.with_overrides()`. CLI callers must omit absent options
from those mappings. Building a new configuration does not mutate the old one.

## Reusable Observations

`save_calibration_samples(samples, path)` atomically saves a supported physical
record as numeric NPZ with a versioned JSON header. `load_calibration_samples()`
validates kind, fields, units and axes before reconstructing its immutable type.
No native files or software are required. Coma response records retain independent
detector-batch means for offline fitting. The format is a sample contract, not
a fitted model; native or historical formats need a downstream importer.

## Fitted-Product Identities

`read_calibration_artifact_identity()` captures the content identity of an
explicit `model.json` and verifies its optional relative `payload` NPZ against
`payload_sha256`. JSON-only products omit both fields. Scientific model loaders
must separately validate their own schema, units, coefficients and support.

`CalibrationArtifactIdentity.verify()` checks every bound file. Orchestration
can persist these identities and reject changed products before execution or
resume, even when paths/settings remain unchanged. A completed result remains
self-contained; analysis need not reopen those external files.
