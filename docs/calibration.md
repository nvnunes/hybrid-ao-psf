# E2E Calibration

Hybrid AO PSF owns the common measurement, filtering, fitting and persistence
operations for reconstructor, covariance, coma and sensing calibration. A
configured downstream `E2ECalibration` subclass supplies native calibration data;
the standalone builders and loaded products require no native E2E software.
Calibration is separate from scientific simulation and never starts implicitly
when a prediction loads its products. The separate concrete
[E2ECalibratedModel](e2e-model.md) composes those products at prediction time.

## Calibration Workflow

Build the four products in this order for a new system. The covariance check
uses the reconstructor prior, and sensing simulation uses both products.

| Calibration | Native input and retained measurements | Offline build and output |
| --- | --- | --- |
| Reconstructor | Measure pre-filter centroid noise on a grid of photon counts and spot metrics. Retain measurements for the grid and interpolation checks. | `build_reconstructor_calibration()` fits a `ReconstructorNoiseModel`. |
| Covariance | Read aligned tilt and coma histories. `measure_covariance_calibration()` filters and measures each zenith plane. | `build_covariance_calibration()` combines the measured planes into `CovarianceFunctions`. The configured build can also check the result against direct loop simulation. |
| Coma | Read paired wavefronts and measure their WFS response. Retain the batch means at each magnitude and zenith plane. | `build_coma_calibration()` fits a joint `ComaResponseModel`. |
| Sensing | Measure zero and signed shifts of fixed WFS spots. Retain the paired draws. | `build_sensing_response()` fits the response and retains noise samples. Simulate command histories with `simulate_sensing_noise()`, then fit the displacement shape with `build_sensing_calibration()` to obtain `SensingModel`. |

Configuration specifies where to save each fitted model. Keep the
measurements for offline rebuilding.

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
ngs_psf = inputs/ngs.pkl
ngs_metrics = inputs/ngs-metrics.pkl

[calibration.covariance]
feedback = [0.2, 0.5, 1.0, 1.5, 2.0]

[calibration.sensing]
displacement = 10 mas

[calibration.reconstructor]
signal_point_count = 129
fwhm_point_count = 97
ee_point_count = 17
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

## Reusable Calibration Data

`save_calibration_data(data, path)` atomically saves typed calibration inputs or
measurements as numeric NPZ with a versioned JSON header. `load_calibration_data()`
validates kind, fields, units and axes before reconstructing its immutable type.
No native files or software are required. Coma response records retain independent
detector-batch means for offline fitting. Use the generic data writer for
native-facing records such as telemetry or raw WFS draws. Use
`save_<family>_measurements()` for the measurement bundles passed to fitters
and `save_<family>_calibration()` for the fitted models used at prediction time.

## Measurement And Building

The instance methods `measure_reconstructor()`, `measure_covariance()`,
`measure_coma()` and `measure_sensing()` return reusable physical measurements.
Covariance yields one ZA plane at a time. Coma retains paired coma/mean WFS
responses, four detector-batch means and noiseless comparisons. Sensing retains
uncentred zero/signed displacement samples. `measure_sensing()` loads the
configured NGS HO PSF interpolator once and measures aligned position and
magnitude rows.
That source file must contain fractions per incident photon: native
normalization and image orientation must be resolved before building it.
The common operation preserves cropped flux and applies no recentering or
unit-flux normalization. The configured precision applies when preparing the
loaded source provider. Its original file and producer precision are retained.

Each family has `save_<family>_measurements()` and
`load_<family>_measurements()` operations. Rebuilding requires these numerical
records, not native sources, setup files or subclass hooks. The standalone
`build_<family>_calibration()` operations return an immutable model and separate
numerical diagnostics; they save nothing implicitly.

Configured `build_reconstructor()`, `build_covariance()`, `build_coma()` and
`build_sensing()` apply the INI settings and return the model and diagnostics
without writing files. Call the matching `save_reconstructor()`,
`save_covariance()`, `save_coma()` or `save_sensing()` explicitly to save that
result to its configured destination:

```python
result = calibration.build_reconstructor(measurements)
calibration.save_reconstructor(result)
```

These save methods return the absolute `model.json` path and save
`diagnostics.json` beside it, identifying the model manifest by SHA-256.
An interrupted save restores both metadata files. Retained numerical
measurements remain untouched. The standalone `save_<family>_calibration()`
functions accept a fitted model and an explicit destination, rather than the
configured build result.

```python
from hybrid_ao_psf import (
    build_reconstructor_calibration,
    load_reconstructor_measurements,
    save_reconstructor_calibration,
)

measurements = load_reconstructor_measurements("measurements.npz")
result = build_reconstructor_calibration(measurements)
save_reconstructor_calibration(result.model, "products/reconstructor/model.json")
```

Covariance filters each realization independently, retains the common settled
window, and uses one pooled realization/time mean with population divisors and
equal realization weighting. It preserves rectangular corner mapping, one
feedback factor per filtered input, airmass interpolation, transpose checks,
whole-realization bootstrap and withheld-position/feedback diagnostics. Hybrid
uses a fixed bootstrap count and seed for the sampling uncertainty report. Different
realization counts across ZA are allowed; directions, frame counts, controller
and settled windows must agree. Pass `validation_telemetry` to
`build_covariance()` to check the calibrated reconstruction for each plane
against direct matrix-loop simulation at the feedback grid points. This
comparison uses the configured reconstructor prior.

The reconstructor prior uses either P×EE and FWHM or separate P, EE and FWHM
axes. The point counts specify the length of each axis. The EE count applies
when EE has its own axis. Hybrid selects the interpolation check points
using a fixed seed.
The build checks isotropy, cross-axis response and the product reduction
against retained native measurements. This prior chooses weights; it is distinct from
measured WFS noise variance. `build_lo_reconstructor()` assembles absolute
field-centre weights from loaded covariance/prior products, with I for one NGS.
`solve_lo_reconstructor()` exposes the same solver for supplied physical
covariance matrices. Neither operation conditions matrices or normalizes blocks.

Coma building estimates scalar/matrix response, bootstrap and batch diagnostics,
then fits the magnitude/airmass law. At least three distinct ZA planes are
required; fitting one new plane alone does not replace the joint model. The
model enforces the measured magnitude and viewing-angle ranges. New measurements
record Hybrid’s fixed bootstrap seed. Both build methods use each plane’s recorded
seed for the acceptance checks. When the data include 30° and at least three
other zenith angles, Hybrid checks the 30° plane against a fit to the others.

## Sensing Noise Simulation And Shape Fitting

Call `statistics()` on a saved sensing measurement after loading it to inspect
its local gain matrix and conditional noise covariance. `build_sensing_response()`
streams retained point measurements into gain/log-variance interpolators and
paired white-noise samples. Deliberate nonuniform magnitude intervals can be
supplied explicitly; unmeasured grid points, gaps and extrapolation are rejected.
This intermediate response is not a complete runtime sensing model.

`simulate_sensing_noise()` accepts the prepared response, physical
`SensingNoiseRequest` cases, the loaded covariance interpolators and calibrated prior.
The configured `simulate_sensing_noise()` method loads those two prerequisites from the
INI. Missing products raise without launching another calibration. Simulation uses
the actual reconstructed feedback matrix and the fixed-gain delayed integrator,
with one pooled mean after settling. `save_sensing_noise()` and
`load_sensing_noise()` retain the histories and numerical inputs for offline
fitting. `noise_realizations` sets the independent histories per case, and
`noise_frames` sets the retained frames after `discard_frames`. Measured v_g supplies loop noise independently of the reconstruction
prior. Retained histories must agree with recorded covariance and means within
the captured history precision and configured reduction roundoff; empirical width need not
equal the analytical stationary width.

`build_sensing_calibration()` fits the covariance-preserving two-Gaussian shape
from all supplied case histories, balancing single- and multiple-NGS families.
It requires simulation coverage at every measured response ZA. Only the completed
response-plus-shape product is ready to save. Runtime width uses the analytical
Lyapunov covariance; the fitted mixture changes shape while preserving it.
`FixedGainIntegrator` owns filtering, state transitions and stationary
noise-command covariance. `NoiseBlurModel.otf()` requires an angular-squared
covariance and angular pixel scale as quantities.

## Product Persistence

A saved model consists of its manifest (`model.json`) and, when needed, a
separate array file (`.npz`). Coma coefficients fit entirely in the manifest.
Each family saves its model using
`save_<family>_calibration()` / `load_<family>_calibration()`. Models have strict
versioned scientific schemas, canonical units, numerical settings and measured
ranges. The numerical table products use
`covariance-interpolators.npz`, `sensing-calibration-interpolators.npz`, and
`reconstructor-noise-interpolator.npz`. Each manifest records the array file’s name and SHA-256 digest. A changed array file gets a numbered filename, such as
`covariance-interpolators-2.npz`, so the previous array file stays available while
the manifest is saved. Saving identical data reuses the current named
array file. Existing manifests still load using their recorded array file filenames.
Serialize writers.
New files respect the process umask and replacement preserves file permissions.

Loaders do not search native directories, infer another product generation,
extrapolate calibrated ranges or use MASTSEL as a fallback. Model state is immutable;
providers can be reused without exposing writable interpolation arrays.

## Model Files

`read_calibration_model_files()` returns the manifest and array-file paths and
checksums for one calibrated model. It verifies the optional relative NPZ
`array_file` against `array_file_sha256`. JSON-only models omit both fields.
Scientific model loaders separately validate schema, units, coefficients and
ranges.

`CalibrationModelFiles.verify()` checks that the manifest and array files still
match their recorded checksums. Orchestration can retain these records and reject
changed files before execution or resume, even when paths and settings are
unchanged. A completed result remains self-contained; analysis need not reopen
those external files.

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
Sensing and covariance calibration records accept controllers from either
package and store them at the record's dtype. A float32 Hybrid record thus
converts the default float64 AO Models controller to float32.
Integer indices, counters, boolean masks and native sampling seeds retain their
nonfloating types. Population divisors, independent resets, equal realization
weights, one pooled mean, settling requirements and feedback normalization are
unchanged. Numerical checks account for dtype and operation scale separately
from physical acceptance criteria.

Fitted products and reusable calibration data use version 2 with a precision
declaration. Version-1 inputs remain readable without imposing the new-work
default. Loaders preserve stored numeric values and do not invent producer
precision for unlabelled fitted models. Version-2 numeric array files must match
their declared dtype. Older noise-history files may contain float32 histories alongside
float64 moments; their captured history precision is retained for quantization
checks.

`CovarianceFunctions`, `ReconstructorNoiseModel`, `ComaResponseModel` and
`SensingModel` offer `prepare(dtype=...)` for immutable runtime conversion.
Their producer-precision record remains unchanged, and the original file
is neither mutated nor rewritten. This conversion does not run calibration.
`grid_precision` separately retains the coarsest coordinate rounding across
conversion and save/load, so promotion keeps the original boundary queries
valid without changing the producer record. It does not permit extrapolation.

The blur fit supplies an analytic gradient to L-BFGS-B, differentiating the
broad component through the unit-covariance constraint. It retains the binned
likelihood, family weights, bounds and four starting points. Callbacks use the
selected precision; convergence status alone is not a distribution-accuracy
check, particularly for nearly Gaussian mixtures at float32.

Sensing model manifests use version 3 and declare `centring = "pooled"`.
Sensing calibration reads and writes the current model and noise-history
formats. INI controls use `noise_realizations` and `noise_frames`.
