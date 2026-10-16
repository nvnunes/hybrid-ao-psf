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
metadata, provenance, and a prepared SciPy regular-grid interpolator.
`save_science_ho_psf_interpolator()` writes a version-2 pickle with identity
`hybrid_ao_psf_science_ho_psf_interpolator`. Loading returns ready-to-evaluate
read-only state without preparing the interpolator again.
PSF pixels use float32 by default. Pass `dtype=np.float64` to
`build_science_ho_psf_interpolator()` when source precision must be retained,
for example for fixed-spot WFS calibration. Both stored grids and provider
predictions preserve the chosen precision; no flux normalization is applied.
The loader accepts existing float32 version-2 artifacts and optional float64
values in the same prepared-array contract. Other pixel precisions are rejected.
`ScienceHoPsfPrediction` also accepts an explicit `dtype` for custom providers.
The simulation engine continues to produce float32 PSF results.

Source images must follow the [coordinate and image orientation contract](architecture.md#coordinate-and-image-orientation);
interpolation does not rotate or flip their pixels.

## NGS High-Order Metrics

`NgsHoMetricSamples` holds already measured EE, FWHM, and Strehl samples.
`NgsHoPsfSamples` plus `build_ngs_ho_metric_samples_from_psfs()` uses AO Stats
to derive those metrics from source PSFs. Regular-grid and scaled-RBF
strategies are selected with `RegularGridInterpolationConfig` and
`RbfInterpolationConfig`.

`build_ngs_ho_metric_interpolator()` produces an immutable
`NgsHoMetricInterpolator`; evaluation returns `NgsHoMetricPrediction`. A fitted
RBF product contains ready-to-evaluate SciPy state that has no simpler portable
array representation. NGS artifacts use version-1 pickles with identity
`hybrid_ao_psf_ngs_ho_metric_interpolator`. Project-owned types in the pickle
use stable `hybrid_ao_psf` module paths. Because pickle reconstruction can
execute Python code, both interpolator families must come from trusted sources.
Interpolation strategy configuration types persist through the public
`hybrid_ao_psf.interpolation.config` module. An immutable wrapper protects the
ready-to-evaluate SciPy RBF state from caller mutation.

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
