# Interpolation Products

Hybrid AO PSF provides interpolators for PSF images and measured PSF metrics.
Both accept samples from science or NGS directions.

## PSF Images

`PsfSamples` holds a PSF cube or a stack of cubes and their physical
metadata. Each cube contains all field points for one wavelength and zenith
angle. Its shape is `(field_points, rows, columns)`. A stack has shape
`(cubes, field_points, rows, columns)`.
`build_psf_interpolator()` produces a read-only
`PsfInterpolator`. Call its `evaluate()` method to obtain a `PsfPrediction`.
Image and metric providers use this same method interface.

```python
from hybrid_ao_psf import build_psf_interpolator

psfs = build_psf_interpolator(samples)
prediction = psfs.evaluate(x=x, y=y, wavelength=wavelength, zenith_angle=zenith_angle)
```

The interpolation data consists of explicit axes, grids and physical metadata.
Evaluation uses a cached SciPy regular-grid evaluator, built on first use or
loaded from the saved file. Repeated calls reuse it without a separate
preparation step.
`save_psf_interpolator()` writes a version-3 pickle with identity
`hybrid_ao_psf_science_ho_psf_interpolator`. Loading returns ready-to-evaluate read-only state.
PSF images and metrics use float32 by default. Pass `dtype=np.float64` to
`build_psf_interpolator()` when source precision must be retained,
for example for fixed-spot WFS calibration. Both stored grids and provider
predictions preserve the chosen precision; no flux normalization is applied.
`PsfPrediction` and `PsfMetricPrediction` also accept an explicit
`dtype` for custom providers. Metric measurement and interpolation accept
the same option, including the `ao_core.stats` measurement step.

Choose precision with `dtype` when building or loading an interpolator.
Loading without `dtype` preserves stored precision. The interpolator retains
`grid_precision`, the coarsest precision used for its coordinates, so coordinate
validation allows rounding at grid boundaries. Hybrid applies
`HybridRequest.dtype` internally without changing the caller's provider.

Source images must follow the [coordinate and image orientation contract](architecture.md#coordinate-and-image-orientation);
interpolation does not rotate or flip their pixels.

## PSF Files

`PsfFileProvider` supplies existing PSFs through the same `PsfProvider`
contract as the image interpolator. Supply a sequence of file paths and a
reader returning `PsfSamples`. The reader owns the file format, physical units,
and image orientation. The example FITS reader is
`read_psf_samples()` in `examples/interpolation/psf_fits.py`.

```python
from pathlib import Path

from hybrid_ao_psf import PsfFileProvider, simulate
from psf_fits import read_psf_samples

science_psfs = PsfFileProvider(
    files=sorted(Path("samples/science-psfs").glob("*.fits")),
    reader=read_psf_samples,
)
result = simulate(request, science_psfs=science_psfs, ngs_metrics=ngs_interpolator)
```

The `psf_fits` import in this example is available when running from
`examples/interpolation`. Each prediction reads the supplied files and returns
saved images matching the wavelength, zenith angle, and field coordinates in
request order. Coordinate comparison allows numerical rounding. A missing or
duplicate match raises an error. All images returned by one call must share
image dimensions, pixel scale, and telescope aperture. The constructor
accepts `dtype`; Hybrid applies its request precision internally.

After running the MASTSEL example, compare file and interpolator predictions
at saved coordinates from the repository root.

```bash
./.conda/bin/python examples/interpolation/compare_psf_sources.py \
  --configuration examples/mastsel/hybrid.ini \
  --science-psfs examples/interpolation/samples/science-psfs \
  --science-interpolator examples/mastsel/products/science.pkl \
  --ngs-metrics examples/mastsel/products/ngs-metrics.pkl
```

The comparison uses one request with both sources and checks the input images,
physical sampling, final Hybrid PSFs, and low-order covariance.

## PSF Metrics

Use `build_psf_metric_interpolator_from_psfs()` to measure EE, FWHM and
Strehl and fit a metric provider in one call. To retain the measurements or
use a specific PSF measurement method, call `measure_psf_metrics()` first. Pass its
`PsfMetricSamples`, or measurements from another source, to
`build_psf_metric_interpolator()`.

The resulting `PsfMetricInterpolator` evaluates field coordinates and, when
sampled, airmass. Wavelength enters the measurement. The builder selects
linear interpolation for a complete rectangular field grid and a scaled
thin-plate-spline RBF for scattered field samples. The RBF uses smoothing 0.05
and a linear polynomial. These numerical choices are fixed; callers supply
samples rather than interpolation settings. The read-only `interpolation_type`
records the selected method. Grids, RBF evaluators and coordinate scaling are
private interpolation data; callers use the provider's `evaluate()` operation.
Duplicate field points or zenith-angle samples are rejected before fitting, and
evaluation outside the sampled region is rejected.

The internal RBF data stores `coordinate_mean` and `coordinate_scale`, the
mean and standard deviation of each coordinate. `varying_axes` selects the
coordinates with nonzero spread. Fitting and evaluation subtract the means
and divide by the scales on those axes.

PSF metric files use version-4 pickles with identity
`hybrid_ao_psf_ngs_ho_metric_interpolator`. They store the selected type, interpolation
data, units and precision. Loading preserves the saved fit; it does not select
another method or refit the samples. RBF files retain ready-to-evaluate SciPy
evaluators protected by a read-only wrapper. Both interpolator families must come
from trusted sources because pickle reconstruction can execute Python code.

## Validation And Comparison

Building and loading validate the interpolation data. Evaluation checks requested
coordinates and rejects values outside the sampled region or physical ranges.
These checks are internal; callers do not run a separate validation operation.
After the range check, small numerical offsets outside the rectangular field
are clipped to its bounds.
`compare_psf_interpolator()` compares predicted images with supplied reference
PSFs and returns a `PsfComparisonSummary` of image and metric errors.
`compare_psf_metric_interpolator()` compares predicted metrics with reference
measurements and returns a `PsfMetricComparisonSummary`. Reference samples may
be source samples or independent samples within the interpolator's range.
Both summaries report `metric_rms_error` and `metric_max_abs_error` in each
metric's units. Image comparisons also report `pixel_scale_max_abs_error` in
milliarcseconds.

Loading performs full contract validation; callers do not need to reconstruct
interpolation data from source samples.

Inputs passed to builders are not retained as mutable storage. Built and loaded
providers own their array and quantity state and expose it read-only, so one
provider may be reused safely across repeated simulation calls.

File writers use sibling temporary files and atomically replace their
destinations only after writing the complete file. An interrupted or failed
overwrite therefore leaves the prior file intact.

The canonical loaders accept the Hybrid AO PSF file identities
and supported versions described above. Conversion from other file formats occurs outside these
loaders.
