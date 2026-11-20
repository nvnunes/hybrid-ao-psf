# Architecture

This document is the source of truth for Hybrid AO PSF package structure,
scientific ownership, public API boundaries, and saved file contracts.

## Project Boundary

Hybrid AO PSF owns:

- PSF image and metric interpolation products;
- translation of Hybrid inputs into MASTSEL configuration;
- explicit MASTSEL and E2E-Calibrated low-order execution;
- a downstream field-wide Ctot hook with transient runtime/statistics context;
- finite-field Gaussian/mixture blur, jitter, and engine-level diagnostics;
- standalone combined-INI and versioned HDF5 result contracts; and
- common E2E calibration and noise-blur kernels, immutable calibrated
  products and the Hybrid controller and reconstructor interfaces.

Hybrid AO PSF does not own batch simulation orchestration, external dataset
schemas, instrument calibration selection, instrument-specific policy, or
final PSF statistics. On the MASTSEL path, callers may supply WFS-effective
photon rates directly or provide magnitudes with an explicit caller-owned
zeropoint. Hybrid converts magnitudes using MASTSEL geometry and AO Core.

## Dependencies

The package uses `ao_core.models` for shared AO configuration, reconstruction,
tip–tilt control and residual covariance assembly. Hybrid keeps its controller and reconstructor
entry points and their float32 defaults. `ao_core.stats` provides PSF metadata
and measurements. `ao_core.photometry` converts magnitudes to effective rates. The low-order
engine supports a tested range of MASTSEL
releases. It has no dependency on an orchestration framework or instrument
package.

## Public API

The `hybrid_ao_psf` package root is the canonical user import boundary. It
re-exports every supported user-facing contract and operation. The
`hybrid_ao_psf.interpolation` submodule is also public for focused
interpolation workflows, and its public names are the same names re-exported
from the package root. Other submodules are implementation details; storage
mechanics and MASTSEL adaptation are private. PSF providers expose `evaluate()`.
Precision conversion stays internal; callers choose `dtype` when building or
loading an interpolator, or on the Hybrid request. The
package-root OTF helpers are the supported numerical primitives for external
finite-field blur comparisons; other numerical helpers remain private.

Physical values use Astropy quantities. Public requests and results are
immutable typed objects. A simulation call accepts an observation request,
a science PSF provider and a low-order model. Orchestration and provider reuse
belong to the caller.

The science provider owns its PSF pixel scale and metadata. The engine uses
that scale for blur and preserves it in the result; it does not relabel or
resample provider output. Concrete providers also encapsulate their fitted
state behind read-only interfaces so reuse cannot be altered through exposed
SciPy objects or writable arrays.

The public contract names are `HybridRequest`, `PsfProvider`,
`PsfMetricProvider`, `HybridResult`, and `HybridDiagnostics`. The `LowOrderModel`
enum identifies the model in configuration and results. `LowOrderDistribution`
records the applied centred Gaussian mixture in angular x/y coordinates. Provider
predictions use `PsfPrediction` and `PsfMetricPrediction`. The
read-only concrete provider types are
`PsfInterpolator` and `PsfMetricInterpolator`. `PsfFileProvider` reads
existing images using a caller-supplied reader returning `PsfSamples`.
The provider owns file access and matches saved coordinates. The stateless
engine accepts either through `science_psfs`. Built and loaded interpolators
are validated, ready to evaluate and reusable. Coordinate checks run inside
evaluation; validation helpers are private. Public comparison operations measure
interpolation errors against reference samples. The engine converts built-in
providers internally when the request requires another precision. The combined
configuration boundary is `HybridConfiguration` plus `load_configuration()`.
The loader uses AO Core's shared physical fields and adds Hybrid's request,
model settings, and provider paths. Standalone runs write results through
`run_configuration()`.

`HybridCtotCorrection` pairs a callable with optional input file paths.
`HybridCtotContext` passes resolved observation and controller inputs with
the read-only baseline quantities in `HybridMastselStatistics`.
The private MASTSEL adapter owns runtime construction, instance-local capture
and fixed-gain noise recovery. It does not modify MASTSEL itself. Correction
models and calibration selection stay with the caller. The engine invokes the
callback before conditioning and its single Gaussian blur. See the [API guide](api.md#covariance-corrections).

`simulate()` borrows its immutable providers, dispatches the supplied
`MastselModel` or `E2ECalibratedModel` and returns a new result.
`simulate_configuration()` constructs the model selected by the INI and runs
it in memory. The CLI uses `run_configuration()`, which saves the result and
records the observation inputs and file paths.
The MASTSEL path reads the scalar LO frame rate from `RTC.SensorFrameRate_LO` in the
effective configuration and expands it to one value per NGS for the native call.
For magnitude requests it also reads `telescope.TelescopeDiameter` and the
first `sensor_LO.NumberLenslets` value, converts once before MASTSEL, and
reports the WFS-effective photon rates, supplied magnitudes and zeropoint.
The standalone `[hybrid]` section supplies magnitudes and their zeropoint.
Hybrid converts them through AO Core photometry.
Hybrid passes the effective configuration in memory on MASTSEL releases
that accept it. Earlier supported releases read the same configuration
through their file interface. Both paths expose the effective INI in debug diagnostics.
The E2E path receives a prepared `E2ECalibratedModel` with its own metrics,
photometry and controller settings. It convolves the science PSFs with the full
calibrated Gaussian mixture; it does not construct MASTSEL or accept its INI,
external metric provider, zeropoint or correction callback. Both paths share
science-provider preparation, physical axes, finite-field FFT/crop and result
contracts. Providers may be reused; simulation does not mutate them.
`blur_psfs_with_ctot()` is a focused operation for externally supplied PSFs
and already-conditioned Ctot; it uses the same finite-field blur without
running MASTSEL or producing a complete simulation result.
`image_plane_gaussian_otf()` and `apply_finite_fov_otfs()` expose the same
Gaussian OTF construction and in-place finite-field application when callers
provide pixel-space covariance or precomputed OTFs directly.

## E2E Calibration Boundary

Hybrid AO PSF owns physical calibration contracts, common configuration,
measurement/build/simulation operations, numerical diagnostics, model persistence
and calibrated model file checks. `E2ECalibration` supplies common workflows with
distinct native reading/measurement hooks. Downstream subclasses own E2E setup,
source compatibility/conversion, normalization, native execution and reuse
checks. No opaque native metadata exchange or adapter registry is required.

Reusable calibration data, model building and model saving remain separate.
Configured build methods return results without writing files; explicit save
methods write those results to the configured destinations. Common code is
shared by Python and native CLI workflows. Loaded calibrated products and offline builders require
neither a native subclass nor native files/software. Numerical kernels have one
upstream owner; calibration does not depend on downstream correction modules.
The four calibration products feed the separate concrete `E2ECalibratedModel`.
It owns the prepared fitted inputs, covariance interpolation, physical
acceptance checks and displacement mixture. AO Core calculates the reconstructor,
feedback, noise propagation and residual covariance from those inputs. Hybrid
also owns the stored diagnostics. Its INI loader exposes
checked model-file paths and checksums without acquisition metadata. It does not
construct science PSFs itself. The canonical engine combines its prediction
with a science provider. See
[Calibrated LO prediction](e2e-model.md) for this runtime boundary. See [Calibration](calibration.md) for lifecycle and scientific contracts.

## Coordinate And Image Orientation

`HybridRequest.science_positions` and `HybridRequest.ngs_positions` are
Cartesian angular arrays with one `(x, y)` pair per row. Hybrid stores the
pairs in arcseconds and passes them to MASTSEL in that order. Each Ctot matrix
uses `(x, y)` on both axes: element `[0, 0]` is x variance, `[1, 1]` is
y variance, and `[0, 1]` is x/y covariance. PSF cubes have shape
`(N, y, x)`. Increasing column index points toward `+x`; increasing row
index points toward `+y` in the same Cartesian frame. The Gaussian OTF maps
x covariance to column frequencies and y covariance to row frequencies.

Science providers must supply source PSFs in this orientation. The PSF
metadata provides pixel scale and physical context but does not encode an
image-axis rotation or flip. Hybrid preserves the source array orientation;
it does not infer or apply a transform from metadata. This contract does not
assign celestial north or east to either Cartesian axis.

## Saved File Contracts

Interpolator files use project-owned schema identities and explicit versions.
Standalone HDF5 results use the shared AO Core format.

PSF image interpolation data uses a version-3 pickle with identity
`hybrid_ao_psf_science_ho_psf_interpolator` and a SciPy regular-grid evaluator.
The image interpolator caches this evaluator for repeated calls, building it on
first use or loading it from the saved file. Changing precision creates a new
interpolator with its own evaluator. Pixel arrays retain float32 by default or
explicit float64; loading validates and preserves that stored precision. PSF metric
state may contain ready-to-evaluate SciPy RBF
objects and uses a version-4 pickle with identity
`hybrid_ao_psf_ngs_ho_metric_interpolator`. Pickle-owned classes resolve only
through stable `hybrid_ao_psf` module paths. Both pickle inputs must be trusted, while all file writers protect existing
destinations with atomic replacement.
Both formats store `grid_precision` alongside the interpolation data. Metric
files use the current format with the selected `interpolation_type` and
interpolation data. Metric builders
select linear interpolation for complete rectangular field grids and a fixed
scaled RBF for scattered samples; callers do not configure numerical methods.

New files respect the process umask, and replacements preserve the
destination's existing mode.

Standalone results store one science field using AO Core’s reader and writer.
Hybrid owns the diagnostic declarations and computed values for each model
and level. The standalone INI workflow
records the observation settings and input paths beside that result.
See [Result files](results.md) for the layout and validation contract.
