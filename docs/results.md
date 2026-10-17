# Result Files

`save_result()` writes one `HybridResult` to HDF5. Files identify themselves
with root attributes `schema = "hybrid-ao-psf-result"`, `schema_version = 3`,
`lo_model = mastsel|e2e-calibrated` and `precision = float32|float64`.
`load_result()` also accepts MASTSEL versions 1 and 2. Version 1 preserves each
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
│   ├── source_metadata/<name>
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
The centred applied distribution stores positive, normalized component weights
and angular covariance matrices of shape `(science, component, 2, 2)` in
physical x/y axes. Their weighted covariance equals angular Ctot. MASTSEL uses
one Gaussian; E2E retains the full calibrated mixture.

MASTSEL results include wavefront Ctot, its angular conversion, effective
`photon / s` flux and effective INI. Magnitudes and caller-supplied zeropoint
are present together for magnitude requests. E2E omits these MASTSEL-only
fields, retains magnitudes without an external zeropoint and stores every
`E2ELowOrderPrediction` field under `e2e_prediction` except dtype and metrics,
which are shared with the parent result. This includes covariance components,
reconstructor, sensing coefficients, feedback decomposition and approximation
facts. No fitted provider or native object is stored. New engine results record
coordinates and ZA; legacy MASTSEL results may lack them.

`provenance` is a string mapping stored as group attributes. E2E execution records
product content identities, controller settings and magnitude convention.
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
