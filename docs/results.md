# Result Files

Hybrid writes one observation in the [AO Core HDF5 format](https://github.com/nvnunes/ao-core).
`save_result()` writes a `HybridResult` and its diagnostics. `run_configuration()`
also records the observation inputs, original INI and resolved file paths.
`load_result()` returns the science result and diagnostics without opening
the referenced inputs.

## Science Output

| Path | Contents |
| --- | --- |
| `/setup/{sci_r,sci_theta}` | Science positions in arcseconds and degrees, in PSF order |
| `/options/wavelength` | Common science wavelength |
| `/meta/{pixel_scale,tel_diameter,tel_pupil}` | Physical PSF metadata with one common pixel scale |
| `/psfs/data` | Final science PSFs with shape `(science, y, x)` |
| `/stats/jitter` | Radial jitter in mas for each science position |

Positions remain Cartesian in Python. The file stores polar coordinates,
with zero angle at the origin. Quantity datasets carry a `units` attribute.
The PSF pixels retain the result precision and the provider's flux convention.

## Run Settings

For an INI run, `/simulation/name` identifies the selected model and
`version` records the Hybrid AO PSF version. The group also holds the original
`base_config` text, effective `precision`, `diagnostics_level`, and MASTSEL's
`non_psd_policy` where applicable.

Resolved paths are stored as `science_ho_psf_interpolator_path` and
`ngs_ho_metric_interpolator_path`. E2E-Calibrated also records
`covariance_model_path`, `coma_model_path`, `sensing_model_path`, and
`reconstructor_model_path`.

INI runs record zenith angle and `ngs_r`, `ngs_theta`, and `ngs_magnitude`
under `/options`. The zeropoint used for MASTSEL magnitudes is recorded
as `/setup/ngs_magnitude_zeropoint`. A supplied magnitude convention is
recorded as `/simulation/ngs_mag_standard`.

## Diagnostics

Hybrid defines the diagnostic fields and AO Core reads and writes them.
`diagnostic_fields()` declares the fields for a model and level.
`diagnostic_values()` returns the corresponding data from one result.

Validation saves `/diagnostics/covariance/ctot` in mas squared and
`psd_valid_mask`. MASTSEL also saves `psd_clipped` and records effective
NGS metrics, flux, scalar `frame_rate`, and the angular-to-wavefront scale
under `/diagnostics/mastsel`.

Debug adds `/diagnostics/psfs_ho` and model-specific details. MASTSEL
saves its effective INI as `/diagnostics/mastsel/effective_ini`.
E2E-Calibrated saves its prediction under `/diagnostics/e2e_calibrated`,
including the displacement mixture weights and angular covariances.

Hybrid checks the scientific relationships among jitter, covariance and
the applied mixture. Saving with explicit overwrite replaces the file atomically.
