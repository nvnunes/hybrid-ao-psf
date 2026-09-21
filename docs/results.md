# Result Files

`save_result()` writes one `HybridResult` to HDF5. Files identify themselves
with root attributes `schema = "hybrid-ao-psf-result"` and
`schema_version = 1`. `load_result()` rejects other identities or versions
rather than guessing at compatibility.

## Layout

```text
/
├── result/
│   ├── psfs
│   ├── metadata/{wavelength,pixel_scale,tel_diameter,tel_pupil}
│   ├── source_metadata/<name>
│   ├── ngs_metrics/{ee,fwhm,sr}
│   ├── ctot/{wavefront,angle,angle_to_wavefront_scale,psd_clipped}
│   ├── ngs/{flux,frame_rate}
│   └── jitter
├── configuration/effective_mastsel_ini
├── provenance
└── diagnostics/  # optional
    ├── psd_valid_mask
    ├── ctot_angle_trace_{min,max,mean}
    ├── ctot_angle_determinant_{min,max,mean}
    └── pre_blur_psfs  # debug only
```

Every physical quantity dataset carries a `unit` attribute containing an
Astropy-readable unit string. The final PSF and optional pre-blur PSF datasets
are dimensionless `float32` arrays in the provider's flux convention. Ctot,
NGS inputs, metrics, jitter, PSF metadata, and source metadata retain explicit
units.

`provenance` is a string mapping stored as group attributes. The effective
MASTSEL INI is stored verbatim as UTF-8 configuration text. Optional
diagnostics occupy their own group. Validation results contain only the PSD
mask and scalar Ctot summaries; debug results additionally contain the large
pre-blur PSF cube. The public diagnostics object derives the PSD-valid count
and fraction from the stored mask.

Saving refuses to replace an existing file unless `overwrite=True` is given.
Writes first complete in a sibling temporary file and atomically replace the
destination, so a failed overwrite preserves the prior result. Loading
reconstructs the validated immutable public result, verifies the relationship
between angular and wavefront Ctot, verifies that jitter is derived from the
angular covariance trace, applies the positive-semidefinite tolerance, verifies
diagnostic masks and summaries against conditioned Ctot, and owns its arrays;
it does not expose mutable HDF5-backed views.
