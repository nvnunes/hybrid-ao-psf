# Calibrated Low-Order Prediction

`E2ECalibratedModel` composes the four fitted calibration products into the
centred LO image-displacement distribution. It predicts covariance components
and the full two-Gaussian distribution parameters without loading native
sources or constructing a MASTSEL model. It does not construct science PSFs;
`simulate()` remains the complete MASTSEL-backed PSF execution path.

## Load And Evaluate

Use the same [shared INI](calibration.md#shared-configuration) as calibration:

```python
from astropy import units as u
from hybrid_ao_psf import load_e2e_calibrated_model

model = load_e2e_calibrated_model("e2e.ini")
result = model.predict(
    ngs_positions=[[12, -24], [-24, 12]] * u.arcsec,
    magnitudes=[15, 17] * u.mag,
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
`product_identities` mapping exposes the four content-bound artifact identities
and the NGS metric file identity for execution/reuse checks. Direct construction
from loaded providers can omit file identities.

## Composition

Saved photometry converts each magnitude to incident photons per frame. The
metric provider supplies NGS HO EE/FWHM/Strehl; the coma and sensing products
supply h, g and v. The separate pre-filter noise prior chooses the absolute
field-centre reconstructor weights from the unfiltered tilt covariance. One
NGS uses identity weights.

The feedback matrix is `B = sum(g_g R_g)`. Its two eigenvalues and projectors
combine the scalar covariance functions without commuting reconstructor blocks
through the loop response. A complex-eigenvalue B uses `(B+B.T)/2` for covariance
bank evaluation, with the approximation flag and relative matrix change in the
result. Defective or excessively ill-conditioned bases raise. The stationary
noise covariance always uses the actual B and the calibrated fixed-gain delayed
integrator through the discrete Lyapunov calculation.

`result.atmospheric` is the atmospheric residual covariance;
`result.ho` contains the HO-command covariance and its cross terms with the
remaining atmospheric motion. The correction can be indefinite. Their sum is
`result.noiseless`; adding `result.noise` gives `result.total`.
Command and science-command covariances remain available for diagnostics.
No covariance component is independently clipped. A noiseless eigenvalue below
-1e-8 mas squared raises; smaller roundoff remains visible.

`mixture_weights` and `noise_scales` describe the covariance-preserving narrow
and broad noise components. `mixture_covariances` adds each scaled noise
covariance to the noiseless covariance, with shape `(science, component, 2, 2)`.
The weighted mixture covariance equals `total`. All covariance axes are physical
x/y and angular covariances are in mas squared; coma response is in mas/nm OPD.

Queries must lie within all applicable spatial, magnitude, zenith and feedback
supports. No extrapolation, native acquisition, MASTSEL fallback or production
PSF integration occurs in this operation. See the [Python reference](reference/api.md)
for the result fields and constructor contracts.
