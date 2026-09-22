# Engine

`simulate()` accepts one resolved `HybridRequest`, one loaded
`SciencePsfProvider`, and one loaded `NgsMetricProvider`. It returns one
immutable `HybridResult` and retains no mutable run state. Loading and provider
reuse remain the caller's responsibility, so fitted products can be reused
efficiently across repeated simulations.

## Scientific Path

One call:

1. evaluates the science-HO-PSF and NGS-HO-metric providers;
2. translates Cartesian request coordinates and resolved physical values into
   an effective MASTSEL INI;
3. calls MASTSEL for Ctot and its angular conversion;
4. validates and, where permitted, conditions Ctot;
5. applies the image-plane Gaussian OTF to a private `float32` PSF copy using
   the provider-owned pixel scale and finite-field convention; and
6. constructs the standard result, jitter, PSD facts, effective configuration,
   provenance, and selected diagnostics.

The finite-field crop is not renormalized. Energy blurred beyond the retained
PSF support remains absent from the result.

For comparisons that supply their own PSFs and conditioned Ctot,
`blur_psfs_with_ctot()` exposes this same finite-field blur without running the
complete simulation. It returns a copy and rejects materially non-PSD Ctot;
callers remain responsible for the covariance conditioning they intend.
Callers with pixel-space covariance or precomputed OTFs can instead use
`image_plane_gaussian_otf()` and `apply_finite_fov_otfs()` directly. The latter
mutates the supplied PSF cube and preserves the same finite-field convention.

## Ctot Policy

Zero Ctot is valid and leaves its PSF unchanged. Ctot is symmetrized before
eigendecomposition. Negative eigenvalues with magnitude no greater than
`1e-9 nm2` are treated as floating-point roundoff and clipped to zero.

The tolerance was fixed after the project-owned released-MASTSEL regression
produced positive eigenvalues from approximately `4.7e3` to `6.9e4 nm2`; it is
many orders of magnitude below the scientific covariance scale. It is an
engine constant, not a researcher setting.

For a materially non-positive-semidefinite matrix, `non_psd_policy = error`
raises before blur and is the default. `non_psd_policy = clip` clips negative
eigenvalues to zero. `HybridResult.psd_clipped` reports every science point
whose covariance required numerical or selected policy clipping.

## Diagnostics

The standard result always retains NGS metrics, Ctot in wavefront and angular
units, angular conversion, resolved NGS inputs, jitter, effective MASTSEL
configuration, and PSD-clipping facts.

`diagnostics_level = none` adds nothing to that standard result.
`diagnostics_level = validation` retains a per-field nonzero-PSD mask, its
count and fraction, and minimum, maximum, and mean angular-Ctot trace and
determinant summaries. `diagnostics_level = debug` retains the same validation
facts plus the potentially large pre-blur PSF cube. Final-PSF EE, FWHM, and
Strehl remain optional AO Stats analyses performed by the caller.
