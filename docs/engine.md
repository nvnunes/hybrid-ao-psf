# Engine

`simulate()` accepts one `HybridRequest`, `science_psfs` implementing
`PsfProvider`, and an explicit `lo_model`. Pass a `MastselModel` with its
NGS metric provider or a loaded `E2ECalibratedModel`. It returns one
immutable `HybridResult` and retains no mutable run state. Loading and provider
reuse remain the caller's responsibility, so fitted products can be reused
efficiently across repeated simulations.

## Scientific Path

For MASTSEL, one call:

1. evaluates the science-HO-PSF and NGS-HO-metric providers;
2. translates Cartesian request coordinates into an effective MASTSEL INI and
   reads the configured LO frame rate;
3. uses supplied effective NGS flux, or converts magnitudes with the
   caller-supplied zeropoint and MASTSEL telescope/WFS geometry;
4. calls MASTSEL for Ctot and its angular conversion;
5. invokes an optional field-wide covariance correction, then validates and,
   where permitted, conditions Ctot;
6. applies the image-plane Gaussian OTF to a private selected-precision PSF copy using
   the provider-owned pixel scale and finite-field convention; and
7. constructs the science result and any requested diagnostics from the
   calculated covariance.

`E2ECalibratedModel.predict()` returns the full low-order displacement
mixture at the requested science positions for the supplied asterism.
The engine converts the angular component covariances to pixel units,
forms the weighted Gaussian OTF and convolves the science PSFs. The
mixture's weighted covariance supplies Ctot and jitter.

Both paths use the provider-owned pixel scale, physical x/y axes and the same
finite-field FFT/crop on a private selected-precision PSF copy. The crop is not renormalized. Energy blurred beyond the retained
PSF support remains absent from the result.

The optional `MastselModel.ctot_correction` receives baseline Ctot and the
resolved numerical inputs for the correction. Hybrid owns the MASTSEL instance
and its temporary configuration through calculation and correction. The
adapter captures covariance and reconstructor returns during the baseline
using instance-local wrappers restored before the callback. Capture is omitted
without a correction definition. Fixed-gain noise is recovered from retained
sensor response/variance using the native sampled transfer function; no detector
moments or atmospheric spectra are recalculated. A missing centre in a multi-NGS
field requires only one additional centre covariance calculation. One NGS needs
no such calculation. See [Covariance corrections](api.md#covariance-corrections)
for the public callback and statistics contracts.

Absent or no-op corrections preserve baseline Ctot. A replacement follows the
same conditioning and single-Gaussian path; Ctot, jitter, PSD facts and diagnostic
summaries all describe that final covariance. The hook returns neither kernels
nor PSFs and adds no second convolution. File paths describe the configured correction.

For comparisons that supply their own PSFs and conditioned Ctot,
`blur_psfs_with_ctot()` exposes this same finite-field blur without running the
complete simulation. It returns a copy and rejects materially non-PSD Ctot;
callers remain responsible for the covariance conditioning they intend.
Callers with pixel-space covariance or precomputed OTFs can instead use
`image_plane_gaussian_otf()` and `apply_finite_fov_otfs()` directly. The latter
mutates the supplied PSF cube and preserves the same finite-field convention.

## Ctot Policy

MASTSEL conditioning applies to its wavefront Ctot. Zero Ctot is valid and leaves its PSF unchanged. Ctot is symmetrized before
eigendecomposition. Negative eigenvalues with magnitude no greater than
`1e-9 nm2` are treated as floating-point roundoff and clipped to zero.

The tolerance is many orders of magnitude below the positive eigenvalues in
the released-MASTSEL regression (`4.7e3` to `6.9e4 nm2`). It is an engine
constant, not a researcher setting.

For a materially non-positive-semidefinite matrix, `non_psd_policy = error`
raises before blur and is the default. `non_psd_policy = clip` clips negative
eigenvalues to zero. `diagnostics.covariance.psd_clipped` reports every science point
whose covariance required numerical or selected policy clipping.

The E2E path uses the validated calibrated distribution unchanged. It rejects
the `clip` policy, validates its covariance components and retains the calibrated distribution. Its checks follow
[Calibrated LO prediction](e2e-model.md#composition).

## Diagnostics

The standard result carries positions, final science PSFs, jitter and
physical PSF metadata. The diagnostics level controls the additional
calculation details available on the result.

- `none` returns the standard result.
- `validation` adds `diagnostics.covariance` for both models. It contains the
  applied angular covariance `ctot` in mas², the `psd_valid_mask` and the
  PSD-valid count and fraction derived from the mask. MASTSEL also supplies
  `covariance.psd_clipped` and a `mastsel` record with `ngs_metrics`, `ngs_flux`,
  `frame_rate` and `angle_to_wavefront_scale`.
- `debug` adds `psfs_ho` for both models. MASTSEL adds `mastsel.effective_ini`.
  E2E-Calibrated adds its full prediction at `e2e_calibrated`, including
  `displacement_distribution`.

Final-PSF EE, FWHM and Strehl are measured with `ao_core.stats`.
