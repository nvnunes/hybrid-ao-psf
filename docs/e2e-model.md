# Calibrated Low-Order Prediction

`E2ECalibratedModel` composes the four fitted calibration products into the
centred LO image-displacement distribution. It predicts covariance components
and the full two-Gaussian distribution parameters without loading native
sources or constructing a MASTSEL model. `model.predict()` supplies standalone LO
statistics; `simulate()` combines the same model with science HO PSFs.

## Load And Evaluate

Use the same [shared INI](calibration.md#shared-configuration) as calibration:

```python
from astropy import units as u
from hybrid_ao_psf import load_e2e_calibrated_model

model = load_e2e_calibrated_model("e2e.ini")
result = model.predict(
    ngs_positions=[[12, -24], [-24, 12]] * u.arcsec,
    ngs_magnitude=[15, 17] * u.mag,
    science_positions=[[0, 0], [12, 12]] * u.arcsec,
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
rounding. Once prepared, prediction performs no file I/O. The read-only
`model_files` mapping records the paths and checksums for the four calibrated
models and the NGS metric file, for execution and reuse checks. Direct construction
from loaded providers can omit these records.

## Composition

Saved photometry converts each magnitude to incident photons per frame.
The prediction reports NGS HO EE, FWHM and Strehl in `ngs_metrics`. In NGS order,
`coma_response_coefficients`, `sensing_gains` and `wfs_noise_variances` give
the fitted coma response h, WFS gain g and per-axis WFS noise variance v.
The model uses `reconstructor_noise_covariance` with the unfiltered tilt
covariance to choose the absolute field-centre reconstructor weights.
One NGS uses identity weights.

`feedback_matrix` stores `B = sum(g_g R_g)`. The model combines the scalar
covariance functions using `interpolation_feedback_eigenvalues` and
`interpolation_feedback_projectors` from `interpolation_feedback_matrix`. The
reconstructor blocks stay inside the loop response. When B has complex
eigenvalues, the model uses `(B+B.T)/2` for interpolation and sets
`used_symmetric_feedback_approximation`. The result also records the relative
change in the feedback matrix and the `eigenbasis_condition_number`. Defective
or excessively ill-conditioned bases raise. The discrete Lyapunov calculation
uses `feedback_matrix` and the calibrated fixed-gain delayed integrator to
compute the stationary noise covariance.

`result.science_tilt_covariance` describes the atmospheric tilt at science
positions before low-order correction. `result.atmospheric_residual_covariance`
describes the atmospheric motion that remains after correction.
`result.ho_correction` combines the HO-command covariance and its cross terms
with the remaining motion. It can be indefinite. Adding it to the atmospheric
residual gives `result.noiseless_covariance`. `result.noise_command_covariance`
gives the covariance of the noise-driven command.
`atmospheric_command_covariance` and `ho_command_covariance` describe the
commands driven by atmospheric tilt and HO-induced apparent tilt, respectively.
`atmospheric_ho_command_covariance` gives the cross-covariance between these
commands. `science_atmospheric_command_covariance` and
`science_ho_command_covariance` give the cross-covariances of atmospheric tilt
at the science positions with each command.
No covariance component is independently clipped. A noiseless eigenvalue below
-1e-8 mas squared raises; smaller roundoff remains visible.

`result.distribution` is a `LowOrderDistribution` with narrow and broad
Gaussian components. The `covariances` array adds the noise covariance scaled
by `noise_covariance_scales` to the noiseless covariance. Its shape is
`(science, component, 2, 2)`.
Use `result.distribution.covariance` for the total image-displacement covariance.
It is the weighted covariance of the Gaussian mixture and agrees with
`result.noiseless_covariance + result.noise_command_covariance` within numerical
precision.
`simulate()` reports the applied angular covariance at `HybridResult.ctot_angle`.
Covariance axes are physical x/y in mas squared.
Coma response is in mas/nm OPD.

Queries must lie within all applicable spatial, magnitude, zenith and feedback
ranges. No extrapolation, native acquisition, MASTSEL fallback or
PSF construction occurs in `model.predict()`. See the [Python reference](reference/api.md)
for the result fields and constructor contracts.

## Complete PSF Execution

Pass the loaded model to the same engine used for MASTSEL:

```python
from hybrid_ao_psf import HybridRequest, load_psf_interpolator, simulate

science = load_psf_interpolator("science.pkl")
request = HybridRequest(
    science_positions=[[0, 0], [12, 12]] * u.arcsec,
    ngs_positions=[[12, -24], [-24, 12]] * u.arcsec,
    ngs_magnitude=[15, 17] * u.mag,
    zenith_angle=20 * u.deg,
    wavelength=2.179 * u.um,
)
result = simulate(request, science, lo_model=model)
```

The supplied model owns the NGS metrics and photometry. The result retains
the applied distribution and fitted LO diagnostics. See [Engine](engine.md),
[Configuration](configuration.md) and [Result files](results.md).
