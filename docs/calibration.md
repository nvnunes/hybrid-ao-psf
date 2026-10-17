# E2E Calibration

Hybrid AO PSF owns the common measurement, filtering, fitting and persistence
operations for reconstructor, covariance, coma and sensing calibration. A
configured downstream `E2ECalibration` subclass supplies native observations;
the standalone builders and loaded products require no native E2E software.
Calibration is separate from scientific simulation and never starts implicitly
when a prediction loads its products. The separate concrete
[E2ECalibratedModel](e2e-model.md) composes those products at prediction time.

## Native Hooks

Subclass `E2ECalibration` and implement its controller/photometry accessors and
five reading/measurement hooks. Covariance reads aligned tilt/coma histories;
coma separates wavefront reading from WFS measurement; sensing measures supplied
spots; reconstructor measures pre-filter centroid-noise covariances. Hooks own
native resources and propagate failures. They preserve means and paired sample
order, and perform no shared LO filtering, response estimation or model fitting.
The common instance measurement methods call these hooks. Native workflows own
source selection and reuse evidence; they call the same common builders as
offline Python workflows. No acquisition metadata enters model evaluation.

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

Paths are relative to the INI, and the four model destinations must be distinct.
Parsing opens neither models nor native sources;
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

## Measurement And Building

The instance methods `measure_reconstructor()`, `measure_covariance()`,
`measure_coma()` and `measure_sensing()` return reusable physical observations.
Covariance yields one ZA plane at a time. Coma retains paired coma/mean WFS
responses, four detector-batch means and noiseless comparisons. Sensing retains
uncentred zero/signed displacement samples. `measure_sensing_point()` accepts
an explicitly prepared `SensingSpot`; `measure_sensing()` loads the configured
NGS HO PSF interpolator once and measures aligned position/magnitude rows.
That source artifact must contain fractions per incident photon: native
normalization and image orientation must be resolved before building it.
The common operation preserves cropped flux and applies no recentering or
unit-flux normalization. The configured precision applies when preparing the
loaded source provider. Its original file and producer precision are retained.

Each family has `save_<family>_measurements()` and
`load_<family>_measurements()` operations. Rebuilding requires these numerical
records, not native sources, setup files or subclass hooks. The standalone
`build_<family>_calibration()` operations return an immutable model and separate
numerical diagnostics; they publish nothing implicitly.

| Family | Reusable observations | Built model |
| --- | --- | --- |
| Reconstructor | Ordered grid, holdouts and equal-signal validation batch | `ReconstructorNoiseModel` |
| Covariance | Six measured covariance tables and per-plane diagnostics | `CovarianceFunctions` |
| Coma | Paired wavefront/WFS response summaries at each magnitude/ZA | `ComaResponseModel` |
| Sensing | Raw WFS samples plus separately retained command replays | `SensingCalibration` |

Configured instance `build_reconstructor()`, `build_covariance()`, `build_coma()`
and `build_sensing()` call these builders and publish to the INI destinations.
They save `diagnostics.json` separately, identifying the corresponding model
manifest by SHA-256. An interrupted diagnostic write can therefore be detected
without making a complete model unusable. Numerical records remain untouched.

```python
from hybrid_ao_psf import (
    build_reconstructor_calibration,
    load_reconstructor_measurements,
    save_reconstructor_calibration,
)

observations = load_reconstructor_measurements("measurements.npz")
result = build_reconstructor_calibration(observations)
save_reconstructor_calibration(result.model, "products/reconstructor/model.json")
```

Covariance filters each realization independently, retains the common settled
window, and uses one pooled realization/time mean with population divisors and
equal realization weighting. It preserves rectangular corner mapping, one
feedback factor per filtered input, airmass interpolation, transpose checks,
whole-realization bootstrap and withheld-position/feedback diagnostics. Different
realization counts across ZA are allowed; directions, frame counts, controller
and settled windows must agree. `validate_covariance_reconstructor()` checks
calibrated reconstruction against independent direct matrix replay at feedback
nodes. Configured covariance building can accept `validation_telemetry` to run
this check for every plane before publication, using the configured prior.

The reconstructor prior supports either P×EE/FWHM or separate P/EE/FWHM axes.
It validates isotropy, cross-axis response and the product reduction against
retained native observations. This prior chooses weights; it is distinct from
measured WFS noise variance. `build_lo_reconstructor()` assembles absolute
field-centre weights from loaded covariance/prior products, with I for one NGS.
`solve_lo_reconstructor()` exposes the same solver for supplied physical
covariance matrices. Neither operation conditions matrices or normalizes blocks.

Coma building estimates scalar/matrix response, bootstrap and batch diagnostics,
then fits the magnitude/airmass law. At least three distinct ZA planes are
required; fitting one new plane alone does not replace the joint model. The
model enforces actual measured magnitude and viewing-angle support. Standalone
building uses recorded bootstrap seeds unless explicitly overridden; configured
building uses the INI/explicit seed without reacquiring WFS samples.

## Sensing Replay And Shape Fitting

`estimate_sensing_statistics()` calculates the local gain matrix and conditional
noise covariance from zero and paired signed draws. `build_sensing_response()`
streams retained point observations into gain/log-variance interpolators and
paired white-noise banks. Deliberate nonuniform magnitude intervals can be
supplied explicitly; unsupported vertices, gaps and extrapolation are rejected.
This intermediate response is not a complete runtime sensing model.

`replay_sensing_calibration()` accepts the prepared response, physical
`SensingReplayRequest` cases, the loaded covariance bank and calibrated prior.
The configured `replay_sensing()` method loads those two prerequisites from the
INI. Missing products raise without launching another calibration. Replay uses
the actual reconstructed feedback matrix and the fixed-gain delayed integrator,
with one pooled mean after settling. `save_sensing_replay()` and
`load_sensing_replay()` retain the histories and numerical inputs for offline
fitting. Measured v_g supplies loop noise independently of the reconstruction
prior. Retained histories must agree with recorded covariance and means within
the captured history precision and configured reduction roundoff; empirical width need not
equal the analytical stationary width.

`build_sensing_calibration()` fits the covariance-preserving two-Gaussian shape
from all supplied case histories, balancing single- and multiple-NGS families.
It requires replay coverage at every measured response ZA. Only the completed
response-plus-shape product is publishable. Runtime width uses the analytical
Lyapunov covariance; the fitted mixture changes shape while preserving it.
`FixedGainIntegrator` owns filtering, state transitions and stationary
noise-command covariance. `NoiseBlurModel.otf()` requires an angular-squared
covariance and angular pixel scale as quantities.

## Product Persistence

Each family publishes an explicit `model.json`, using
`save_<family>_calibration()` / `load_<family>_calibration()`. Models have strict
versioned scientific schemas, canonical units, numerical settings and measured
support. Coma is JSON only; numerical table products reference a relative NPZ
payload and its digest. Payloads are content-named and the manifest is published
last. Interrupted replacement leaves a complete product usable; old payloads
remain available for readers bound to the previous manifest. Serialize writers.
New files respect the process umask and replacement preserves file permissions.

Loaders do not search native directories, infer another product generation,
extrapolate support or use MASTSEL as a fallback. Model state is immutable;
providers can be reused without exposing writable interpolation arrays.

## Fitted-Product Identities

`read_calibration_artifact_identity()` captures the content identity of an
explicit `model.json` and verifies its optional relative `payload` NPZ against
`payload_sha256`. JSON-only products omit both fields. Scientific model loaders
must separately validate their own schema, units, coefficients and support.

`CalibrationArtifactIdentity.verify()` checks every bound file. Orchestration
can persist these identities and reject changed products before execution or
resume, even when paths/settings remain unchanged. A completed result remains
self-contained; analysis need not reopen those external files.

## Numerical Precision

New calibration work defaults to float32. Select `precision = float64` in
`[e2e]` or override the parsed configuration with `with_overrides(dtype=...)`.
The configured workflows resolve this once and pass it into the shared
calculations and native conversion boundaries. The native WFS keeps its own
arithmetic. SciPy interpolation, fitting and Lyapunov solver internals may use
their required precision; callbacks, shared calculations and returned state use
the selected precision.

Standalone measurement/build operations and physical records accept keyword-only
`dtype=np.float32` or `np.float64`, including ordinary NumPy dtype/string
spellings. Other precisions are rejected. Use the same dtype for source records
and operations: promotion cannot recover values already rounded by a reader.
Integer indices, counters, boolean masks and native sampling seeds retain their
nonfloating types. Population divisors, independent resets, equal realization
weights, one pooled mean, settling requirements and feedback normalization are
unchanged. Numerical checks account for dtype and operation scale separately
from physical acceptance criteria.

Fitted products and reusable observations use version 2 with a precision
declaration. Version-1 inputs remain readable without imposing the new-work
default. Loaders preserve stored numeric values and do not invent producer
precision for unlabelled fitted models. Version-2 numeric payloads must match
their declared dtype. Old replay files may contain float32 histories alongside
float64 moments; their captured history precision is retained for quantization
checks.

`CovarianceFunctions`, `ReconstructorNoiseModel`, `ComaResponseModel` and
`SensingCalibration` offer `prepare(dtype=...)` for immutable runtime conversion.
Their producer-precision record remains unchanged, and the original artifact
is neither mutated nor rewritten. This conversion does not run calibration.
`support_precision` separately retains the coarsest coordinate rounding across
conversion and save/load, so promotion keeps the original boundary queries
valid without changing the producer record. It does not permit extrapolation.

The blur fit supplies an analytic gradient to L-BFGS-B, differentiating the
broad component through the unit-covariance constraint. It retains the binned
likelihood, family weights, bounds and four starting points. Callbacks use the
selected precision; convergence status alone is not a distribution-accuracy
check, particularly for nearly Gaussian mixtures at float32.
