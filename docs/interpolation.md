# Interpolation Products

Hybrid AO PSF owns two reusable loaded-provider families. Both preserve the
field, wavelength, zenith/airmass, metric, and replay contracts while using
project-owned persisted identities.

## Science High-Order PSFs

`ScienceHoPsfSamples` holds source PSFs and their physical metadata.
`build_science_ho_psf_interpolator()` produces an immutable
`ScienceHoPsfInterpolator`; `evaluate_science_ho_psf_interpolator()` returns
`ScienceHoPsfPrediction`. The provider's `predict()` method is the equivalent
engine-facing operation.

Science fitted state consists of explicit axes, grids, quantities, source
metadata, and provenance. `save_science_ho_psf_interpolator()` therefore writes
HDF5 with identity `hybrid_ao_psf_science_ho_psf_interpolator` and schema
version 1. `load_science_ho_psf_interpolator()` accepts only that current
project-owned contract and returns ready-to-evaluate read-only state.

## NGS High-Order Metrics

`NgsHoMetricSamples` holds already measured EE, FWHM, and Strehl samples.
`NgsHoPsfSamples` plus `build_ngs_ho_metric_samples_from_psfs()` uses AO Stats
to derive those metrics from source PSFs. Regular-grid and scaled-RBF
strategies are selected with `RegularGridInterpolationConfig` and
`RbfInterpolationConfig`.

`build_ngs_ho_metric_interpolator()` produces an immutable
`NgsHoMetricInterpolator`; evaluation returns `NgsHoMetricPrediction`. A fitted
RBF product contains ready-to-evaluate SciPy state that has no simpler portable
array representation. NGS artifacts therefore remain versioned pickles, now
owned by `hybrid_ao_psf_ngs_ho_metric_interpolator`, version 1. Any project
types in the pickle live at stable `hybrid_ao_psf` module paths. Because pickle
reconstruction can execute Python code, NGS artifacts must come from a trusted
source. Interpolation strategy configuration types persist through the public
`hybrid_ao_psf.interpolation.config` module rather than a private implementation
module. Fitted SciPy RBF state remains ready to evaluate but is held behind an
immutable wrapper instead of being exposed for caller mutation.

## Validation And Replay

Both families provide query and artifact validation before evaluation. Replay
operations compare a provider with its source samples and report interpolation
residual summaries. Loading performs full contract validation; callers do not
need to reconstruct fitted state from training samples.

Inputs passed to builders are not retained as mutable storage. Built and loaded
providers own their array and quantity state and expose it read-only, so one
provider may be reused safely across repeated simulation calls.

Artifact writers use sibling temporary files and atomically replace their
destinations only after successful serialization. An interrupted or failed
overwrite therefore leaves the prior artifact intact.

The canonical loaders accept only the current Hybrid AO PSF artifact identities
and versions. Conversion from other artifact formats occurs outside these
loaders.
