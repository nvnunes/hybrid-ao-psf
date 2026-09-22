# Architecture

This document is the source of truth for Hybrid AO PSF package structure,
scientific ownership, public API boundaries, and persisted contracts.

## Project Boundary

Hybrid AO PSF owns:

- science-HO-PSF and NGS-HO-metric interpolation products;
- translation of resolved Hybrid inputs into MASTSEL configuration;
- MASTSEL low-order execution and Ctot interpretation;
- finite-field Ctot blur, jitter, and engine-level diagnostics; and
- standalone combined-INI and versioned HDF5 result contracts.

Hybrid AO PSF does not own batch simulation orchestration, external dataset
schemas, instrument photometry, magnitude conversion, instrument-specific
policy, or final PSF statistics. Callers resolve those concerns before
constructing an engine request.

## Dependencies

The package depends directly on AO Stats for neutral PSF metadata and the
measurements required by interpolation products. It depends directly on a
tested released MASTSEL range for low-order calculation. It has no dependency
on an orchestration framework or instrument package.

## Public API

The `hybrid_ao_psf` package root is the canonical user import boundary. It
re-exports every supported user-facing contract and operation. The
`hybrid_ao_psf.interpolation` submodule is also public for focused
interpolation workflows, and its public names are the same names re-exported
from the package root. Other submodules are implementation details; storage
mechanics, MASTSEL adaptation, and runtime preparation are private. The
package-root OTF helpers are the supported numerical primitives for external
finite-field blur comparisons; other numerical helpers remain private.

Physical values use Astropy quantities. Public requests and results are
immutable typed objects. One simulation call accepts one resolved request and
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
`ScienceHoPsfInterpolator` and `NgsHoMetricInterpolator`; loading and runtime
preparation remain outside the stateless engine operation. The combined
configuration boundary is `HybridConfiguration` plus `load_configuration()`;
standalone results use `save_result()` and `load_result()`.

`simulate()` is the sole complete scientific simulation path. It is stateless,
borrows its immutable providers, owns MASTSEL translation and execution, and
returns a new result. `simulate_configuration()`, `run_configuration()`, and the
CLI call this operation rather than defining separate scientific behavior.
The engine reads the active NGS frame rate from `RTC.SensorFrameRate_LO` in the
effective MASTSEL configuration and reports the applied rates in the result.
`blur_psfs_with_ctot()` is a focused operation for externally supplied PSFs
and already-conditioned Ctot; it uses the same finite-field blur without
running MASTSEL or producing a complete simulation result.
`image_plane_gaussian_otf()` and `apply_finite_fov_otfs()` expose the same
Gaussian OTF construction and in-place finite-field application when callers
provide pixel-space covariance or precomputed OTFs directly.

## Persisted Contracts

Interpolator artifacts and standalone HDF5 results use project-owned schema
identities and explicit versions. Loaders accept only the current supported
Hybrid AO PSF contracts. Conversion from other artifact formats occurs outside
the canonical loaders.

Science-HO-PSF fitted state uses a version-2 pickle with identity
`hybrid_ao_psf_science_ho_psf_interpolator` and a prepared SciPy regular-grid
interpolator. NGS-HO-metric state may contain ready-to-evaluate SciPy RBF
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
Its root schema identity is `hybrid-ao-psf-result`, currently at version 1.
