# Architecture

This document is the source of truth for Hybrid AO PSF package structure,
scientific ownership, public API boundaries, and persisted contracts.

## Project Boundary

Hybrid AO PSF owns:

- science-HO-PSF and NGS-HO-metric interpolation products;
- translation of Hybrid inputs into MASTSEL configuration;
- MASTSEL low-order execution and Ctot interpretation;
- a downstream field-wide Ctot hook with transient runtime/statistics context;
- finite-field Ctot blur, jitter, and engine-level diagnostics;
- standalone combined-INI and versioned HDF5 result contracts; and
- common E2E calibration, fixed-integrator/reconstruction/noise-blur kernels and
  immutable calibrated products.

Hybrid AO PSF does not own batch simulation orchestration, external dataset
schemas, instrument calibration selection, instrument-specific policy, or
final PSF statistics. Callers may supply effective NGS flux directly or
provide magnitudes with an explicit caller-owned zeropoint. Hybrid converts
the latter to WFS-effective rates using MASTSEL geometry and NGS Photometry.

## Dependencies

The package depends directly on AO Stats for neutral PSF metadata and the
measurements required by interpolation products, on NGS Photometry for
magnitude-to-effective-rate conversion, and on a tested released MASTSEL range
for low-order calculation. It has no dependency
on an orchestration framework or instrument package.

## Public API

The `hybrid_ao_psf` package root is the canonical user import boundary. It
re-exports every supported user-facing contract and operation. The
`hybrid_ao_psf.interpolation` submodule is also public for focused
interpolation workflows, and its public names are the same names re-exported
from the package root. Other submodules are implementation details; storage
mechanics and MASTSEL adaptation are private. Concrete fitted providers expose
immutable public `prepare(dtype=...)` conversion. The
package-root OTF helpers are the supported numerical primitives for external
finite-field blur comparisons; other numerical helpers remain private.

Physical values use Astropy quantities. Public requests and results are
immutable typed objects. One simulation call accepts one request and
loaded interpolation providers and returns one result; orchestration and
provider reuse belong to the caller.

The science provider owns its PSF pixel scale and metadata. The engine uses
that scale for blur and preserves it in the result; it does not relabel or
resample provider output. Concrete providers also encapsulate their fitted
state behind immutable interfaces so reuse cannot be altered through exposed
SciPy objects or writable arrays.

The public contract names are `HybridRequest`, `SciencePsfProvider`,
`NgsMetricProvider`, `HybridResult`, and `HybridDiagnostics`. Provider
predictions use `ScienceHoPsfPrediction` and `NgsHoMetricPrediction`. The
immutable concrete provider types are
`ScienceHoPsfInterpolator` and `NgsHoMetricInterpolator`; file loading remains outside
the stateless engine. The engine prepares concrete providers at request precision
when needed; callers may prepare and reuse them beforehand. The combined
configuration boundary is `HybridConfiguration` plus `load_configuration()`;
standalone results use `save_result()` and `load_result()`.

`HybridCtotCorrection` pairs a callable with immutable static provenance.
`HybridCtotContext` borrows the live runtime and resolved inputs;
`HybridMastselStatistics` exposes read-only mathematical baseline quantities.
The private MASTSEL adapter owns runtime construction, instance-local capture
and fixed-gain noise recovery. It does not modify MASTSEL itself. Correction
models and calibration selection stay with the caller. The engine invokes the
callback before conditioning and its single Gaussian blur; it persists only the
supplied static provenance. See the [API guide](api.md#covariance-corrections).

`simulate()` is the sole complete scientific simulation path. It is stateless,
borrows its immutable providers, owns MASTSEL translation and execution, and
returns a new result. `simulate_configuration()`, `run_configuration()`, and the
CLI call this operation rather than defining separate scientific behavior.
The engine reads the active NGS frame rate from `RTC.SensorFrameRate_LO` in the
effective MASTSEL configuration and reports the applied rates in the result.
For magnitude requests it also reads `telescope.TelescopeDiameter` and the
first `sensor_LO.NumberLenslets` value, converts once before MASTSEL, and
reports both the effective flux and the supplied magnitudes and zeropoint.
The MASTSEL INI contains no instrument zeropoint; the standalone command
supplies resolved flux through its `[hybrid]` section.
`blur_psfs_with_ctot()` is a focused operation for externally supplied PSFs
and already-conditioned Ctot; it uses the same finite-field blur without
running MASTSEL or producing a complete simulation result.
`image_plane_gaussian_otf()` and `apply_finite_fov_otfs()` expose the same
Gaussian OTF construction and in-place finite-field application when callers
provide pixel-space covariance or precomputed OTFs directly.

## E2E Calibration Boundary

Hybrid AO PSF owns physical calibration contracts, common configuration,
measurement/build/replay operations, numerical diagnostics, model persistence
and fitted-product identities. `E2ECalibration` supplies common workflows with
distinct native reading/measurement hooks. Downstream subclasses own E2E setup,
source compatibility/conversion, normalization, native execution and reuse
checks. No opaque native metadata exchange or adapter registry is required.

Reusable observations, standalone builders, explicit save/load operations and
configured publication remain separate. Common code is shared by Python and
native CLI workflows. Loaded calibrated products and offline builders require
neither a native subclass nor native files/software. Numerical kernels have one
upstream owner; calibration does not depend on downstream correction modules.
The four calibration products feed the separate concrete `E2ECalibratedModel`.
It owns immutable prepared providers and standalone scientific LO composition,
including reconstructor weights, scalar-bank assembly, actual-feedback noise
propagation and displacement-mixture parameters. Its INI loader exposes
validated product-content identities without acquisition metadata. It does not
construct science PSFs or alter the MASTSEL-backed engine. See
[Calibrated LO prediction](e2e-model.md) for this runtime boundary. See [Calibration](calibration.md) for lifecycle and scientific contracts.

## Coordinate And Image Orientation

Request and field coordinates are Cartesian `(x, y)` pairs in arcseconds.
Hybrid passes science and NGS pairs to MASTSEL in that order. Each Ctot matrix
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

## Persisted Contracts

Interpolator artifacts and standalone HDF5 results use project-owned schema
identities and explicit versions. Loaders accept only the current supported
Hybrid AO PSF contracts. Conversion from other artifact formats occurs outside
the canonical loaders.

Science-HO-PSF fitted state uses a version-2 pickle with identity
`hybrid_ao_psf_science_ho_psf_interpolator` and a prepared SciPy regular-grid
interpolator. Its prepared pixel arrays retain float32 by default or explicit
float64; loading validates and preserves that stored precision. NGS-HO-metric
state may contain ready-to-evaluate SciPy RBF
objects and uses a version-1 pickle with identity
`hybrid_ao_psf_ngs_ho_metric_interpolator`. Pickle-owned classes resolve only
through stable `hybrid_ao_psf` module paths; persisted interpolation
configuration types live in `hybrid_ao_psf.interpolation.config`. Both pickle
inputs must be trusted, while all artifact writers protect existing
destinations with atomic replacement.
New artifacts respect the process umask, and replacements preserve the
destination's existing mode.

The standalone result keeps quantities, provenance, effective configuration,
and optional diagnostics together without imposing a batch dataset structure.
Its root schema identity is `hybrid-ao-psf-result`, currently at version 2. It records the effective precision. Version-1 mixed
arrays load at their stored precision without a uniform arithmetic claim.
