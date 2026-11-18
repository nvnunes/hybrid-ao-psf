# Result Files

`save_result()` writes one `HybridResult` to HDF5. Files identify themselves
with root attributes `schema = "hybrid-ao-psf-result"`, `schema_version = 4`,
`lo_model = mastsel|e2e-calibrated` and `precision = float32|float64`.
`load_result()` also accepts version 3 and MASTSEL versions 1 and 2. Version 1 preserves each
stored array and exposes `result.dtype=None` because no uniform precision was
recorded. Saving a legacy mixed result retains its arrays and records
`precision=mixed`. Unsupported identities/versions and declared precision
mismatches are rejected.

## Layout

```text
/
├── result/
│   ├── psfs
│   ├── metadata/{wavelength,pixel_scale,tel_diameter,tel_pupil}
│   ├── ngs_metrics/{ee,fwhm,sr}
│   ├── ctot/{angle,psd_clipped,wavefront?,angle_to_wavefront_scale?}
│   ├── ngs/{frame_rate,flux?,magnitude?,magnitude_zeropoint?}
│   ├── coordinates/{science_positions?,ngs_positions?,zenith_angle?}
│   ├── lo_distribution/{weights,covariances}
│   ├── e2e_prediction/  # E2E only
│   └── jitter
├── configuration/effective_mastsel_ini?  # MASTSEL only
├── provenance
└── diagnostics/  # optional
    ├── psd_valid_mask
    ├── ctot_angle_trace_{min,max,mean}
    ├── ctot_angle_determinant_{min,max,mean}
    └── pre_blur_psfs  # debug only
```

Every physical quantity dataset carries an Astropy-readable `unit` attribute.
The PSF arrays retain the result precision and the provider's flux convention.
The `lo_distribution` group stores the centred `HybridResult.distribution`
with positive, normalized component weights and angular covariance matrices
of shape `(science, component, 2, 2)` in physical x/y axes. Their weighted
covariance equals angular Ctot. MASTSEL uses one Gaussian. E2E retains the full
calibrated mixture.

MASTSEL results include wavefront Ctot, its angular conversion, effective
`photon / s` flux and effective INI. Magnitudes and caller-supplied zeropoint
are present together for magnitude requests. E2E omits these MASTSEL-only
fields and retains magnitudes. The `e2e_prediction` group stores the calibrated
covariance components, reconstructor and feedback diagnostics. The `gains` and
`noise_variances` datasets store `sensing_gains` and `wfs_noise_variances`.
The `feedback` and `effective_feedback` datasets store `feedback_matrix` and
`interpolation_feedback_matrix`, respectively.
The `strengths` and `projectors` datasets store `interpolation_feedback_eigenvalues`
and `interpolation_feedback_projectors`. The `symmetric_feedback_fallback` and
`eigenbasis_condition` attributes store `used_symmetric_feedback_approximation`
and `eigenbasis_condition_number`. The `coma_response` and `noise_scales`
datasets store `coma_response_coefficients` and `noise_covariance_scales`.
The `noise_prior` dataset stores `reconstructor_noise_covariance`.
The `atmospheric_command` and `ho_command` datasets store
`atmospheric_command_covariance` and `ho_command_covariance`, respectively.
The cross-covariance datasets `atmospheric_ho_command`,
`science_atmospheric_command` and `science_ho_command` use the corresponding
Python field names with the `_covariance` suffix omitted.
The `uncorrected` and `noiseless` datasets store
`science_tilt_covariance` and `noiseless_covariance`. The `atmospheric`, `ho` and
`noise` datasets store `atmospheric_residual_covariance`, `ho_correction` and
`noise_command_covariance`, respectively. The group stores the prediction
distribution as `mixture_weights` and `mixture_covariances`. The `total` dataset
stores the sum of the noiseless and noise covariances for validation. Loading
checks the mixture against `lo_distribution` and uses the parent result's
precision and `ngs_metrics`.
New engine results record coordinates and ZA. Legacy MASTSEL results may lack
these fields.

`provenance` is a string mapping stored as group attributes. E2E execution records
calibrated model file paths and checksums, controller settings and magnitude convention.
Optional diagnostics contain the PSD mask and scalar Ctot summaries; debug
results also contain the pre-blur PSFs. The public diagnostics object derives
its PSD-valid count and fraction from the mask.

Saving refuses to replace an existing file unless `overwrite=True`. Writes
complete in a sibling temporary file and atomically replace the destination,
so a failed overwrite preserves the prior result. Loading needs no fitted
products, native files or model construction. It reconstructs validated immutable
objects, checks model-specific fields, precision, applied distribution,
component/noise/total covariance relationships, jitter and diagnostic summaries.
MASTSEL results also validate angular/wavefront conversion. Arrays are owned;
mutable HDF5-backed views are not exposed.
