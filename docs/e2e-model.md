# Calibrated Low-Order Prediction

`E2ECalibratedModel` composes the four fitted calibration products into the
centred LO image-displacement distribution. It predicts covariance components
and the full two-Gaussian distribution parameters without loading native
sources or constructing a MASTSEL model. `model.predict()` supplies standalone LO
statistics; `simulate()` combines the same model with science HO PSFs.

## Load And Evaluate

Build the products for the [executable example](configuration.md#executable-e2e-calibrated-example)
and run this from the repository root.

```python
from astropy import units as u
from hybrid_ao_psf import load_e2e_calibrated_model

model = load_e2e_calibrated_model("examples/e2e-calibrated/hybrid.ini")
result = model.predict(
    ngs_positions=[[30, 0], [-15, 25.98076211353316]] * u.arcsec,
    ngs_magnitude=[15, 16] * u.mag,
    science_positions=[[0, 0], [36, 0]] * u.arcsec,
    zenith_angle=20 * u.deg,
)
```

The loader opens the four `model.json` products and `ngs_metrics`. It does not
open `science_psf`, `ngs_psf` or native acquisition references. Missing products
and conflicting controller or photometry settings raise. Calibration controls
in the INI govern calibration, not reinterpretation of fitted products.

Precision resolves from float32 default, then `[e2e] precision`, then an explicit
`dtype` argument. `model.prepare(dtype=...)` converts immutable runtime state
without changing saved products or recovering information lost during earlier
rounding. Once prepared, prediction performs no file I/O. The loader verifies
the calibrated model files and NGS metric interpolator. Standalone INI runs record their manifest and interpolator paths under
`/simulation`. See [Result files](results.md#run-settings).

## Composition

Saved photometry converts each magnitude to incident photons per frame.
Hybrid evaluates the fitted coma response, sensing gain and noise variance,
and NGS spot metrics. The reconstructor uses the atmospheric covariances and
the fitted pre-filter measurement-noise covariance. For one NGS, the
reconstructor is the identity.

The gain-weighted reconstructor determines the actual feedback matrix.
The scalar decomposition determines which filtered covariance functions to
evaluate. When needed, the interpolation uses the symmetric approximation
to the feedback matrix. The discrete Lyapunov calculation uses the actual
feedback matrix to propagate WFS noise.

The atmospheric residual and HO correction add to the noiseless covariance.
The displacement mixture adds a scaled noise-command covariance to each
Gaussian component. A noiseless eigenvalue below `-1e-8 mas2` raises.

## Prediction Result

`predict()` returns a `displacement_distribution` with narrow and broad
Gaussian components. The covariance array has shape
`(science, component, 2, 2)`. Use `prediction.displacement_distribution.covariance`
for the weighted total image-displacement covariance.

With `diagnostics_level="debug"`, the prediction also retains:

- `diagnostics.ngs` with photon counts, spot metrics, coma coefficients, sensing
  gain and variance, and the reconstructor noise prior.
- `diagnostics.control` with the reconstructor, actual feedback and
  `scalar_feedback`. The latter contains the effective feedback, strengths and
  projectors used for covariance interpolation.
- `diagnostics.covariances` with the science-tilt and command covariances in
  `terms`, the atmospheric residual, HO correction, noiseless covariance and
  noise-command covariance. The HO correction can be indefinite.
- `diagnostics.noise_covariance_scales` with the narrow and broad noise multipliers.

All angular covariances use physical x/y axes and mas squared. Coma response
uses mas per nm OPD. `simulate()` reports angular covariance at
`result.diagnostics.covariance.ctot` at Validation and Debug levels. Debug
also exposes the prediction at `result.diagnostics.e2e_calibrated`.

Queries must lie within all applicable spatial, magnitude, zenith and feedback
ranges. No extrapolation, native acquisition, MASTSEL fallback or
PSF construction occurs in `model.predict()`. See the [Python reference](reference/api.md)
for the result fields and constructor contracts.

## Complete PSF Execution

Pass the loaded model to the same engine used for MASTSEL:

```python
from hybrid_ao_psf import load_configuration, load_psf_interpolator, simulate

configuration = load_configuration("examples/e2e-calibrated/hybrid.ini")
science = load_psf_interpolator(configuration.science_psf_file)
result = simulate(configuration.request, science, lo_model=model)
```

The supplied model owns the NGS metrics and photometry. At debug level,
the result retains the applied distribution and the intermediate quantities
used to calculate it. See [Engine](engine.md), [Configuration](configuration.md)
and [Result files](results.md).
