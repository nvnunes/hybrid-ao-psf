# Configuration

Hybrid AO PSF uses one INI file for one simulation. `[hybrid] lo_model` selects
`mastsel` (the default) or `e2e-calibrated`. Paths resolve relative to that file.

## MASTSEL Configuration

Existing MASTSEL sections retain their familiar names and fields. The separate `[hybrid]` section owns
artifact selection, standalone output, resolved high-order inputs, and engine
policy.

### Hybrid Section

The `[hybrid]` section requires:

| Field | Meaning |
| --- | --- |
| `science_psf_artifact` | Loaded science-HO-PSF artifact path |
| `ngs_metric_artifact` | Loaded NGS-HO-metric artifact path |
| `output` | Standalone HDF5 result path |
| `ngs_flux_photon_per_s` | One resolved photon-rate value per active NGS |

Artifact and output paths are resolved relative to the INI file. The artifact
paths must exist when the configuration is loaded; the output and its parent
directory may be created later.

`precision = float32|float64` selects shared numerical state, calculations and
outputs; the default is float32. Python `load_configuration(..., dtype=...)` or
`run_configuration(..., dtype=...)` and CLI `--precision` override the INI.
Omitting an override preserves the INI choice. `HybridRequest.dtype` supplies
the same boundary to direct Python calls. Loaded concrete providers are
prepared once for that precision, without changing their saved producer state.
MASTSEL and native WFS arithmetic remain engine-owned; SciPy internal solver
workspace may differ, while shared callbacks use the selected dtype.

`non_psd_policy` is optional and defaults to `error`. The other accepted value
is `clip`. `diagnostics_level` is optional and defaults to `none`. The
`validation` level retains lightweight PSD validity and angular-Ctot summary
facts. The `debug` level retains those facts plus the pre-blur PSF cube.

The loaded science-PSF artifact owns the PSF pixel scale. Hybrid AO PSF uses
that scale for Ctot blur and preserves it in the result metadata; it does not
relabel or resample the provider output. The retired `pixel_scale_mas` Hybrid
option is rejected so a stale configuration cannot silently imply otherwise.

### MASTSEL-Owned Inputs

The loader reads the following MASTSEL fields to construct the canonical
request:

| Section and field | Request value |
| --- | --- |
| `telescope.ZenithAngle` | Scalar zenith angle in degrees |
| `sources_science.Wavelength` | One science wavelength in metres |
| `sources_science.Zenith`, `Azimuth` | Science polar coordinates in arcseconds and degrees |
| `sources_LO.Zenith`, `Azimuth` | Active-NGS polar coordinates in arcseconds and degrees |

The two polar coordinate pairs are converted to Cartesian arcsecond vectors
before `HybridRequest` is constructed. All other non-Hybrid sections and
values remain MASTSEL-owned configuration and are retained for engine use.
The engine reads the positive scalar `RTC.SensorFrameRate_LO` from the effective
MASTSEL configuration and applies it to each active NGS.

### Example Shape

```ini
[telescope]
ZenithAngle = 30.0

[sources_science]
Wavelength = [1.65e-6]
Zenith = [0.0]
Azimuth = [0.0]

[sources_LO]
Wavelength = [710e-9]
Zenith = [30.0, 30.0, 30.0]
Azimuth = [0.0, 120.0, 240.0]

[RTC]
SensorFrameRate_LO = 500.0

[hybrid]
science_psf_artifact = products/science.pkl
ngs_metric_artifact = products/ngs.pkl
output = results/hybrid.h5
ngs_flux_photon_per_s = [1200.0, 950.0, 1100.0]
non_psd_policy = error
diagnostics_level = none
```

This abbreviated example omits other MASTSEL sections required by real engine
execution. Hybrid AO PSF passes those sections through its owned translation
boundary rather than reclassifying them as Hybrid fields.

### Executable Sample

The repository's `examples/hybrid.ini` contains a complete small MASTSEL
configuration. From a bootstrapped checkout:

```bash
./.conda/bin/python examples/build_sample_artifacts.py
./.conda/bin/hybrid-ao-psf examples/hybrid.ini
```

The builder creates the two ready-to-evaluate products referenced by the INI.
The command writes the configured `examples/results/hybrid.h5` result. The
sample is also the Python-and-CLI real-MASTSEL end-to-end verification route.

## E2E-Calibrated Configuration

Use the same `[e2e]` model/provider paths as
[calibration](calibration.md#shared-configuration). Add Cartesian prediction
inputs in `[hybrid]`:

```ini
[e2e]
precision = float32
science_psf = products/science.pkl
ngs_psf = products/ngs-psf.pkl
ngs_metrics = products/ngs-metrics.pkl
reconstructor_model = products/reconstructor/model.json
covariance_model = products/covariance/model.json
coma_model = products/coma/model.json
sensing_model = products/sensing/model.json

[hybrid]
lo_model = e2e-calibrated
output = results/hybrid.h5
science_x_arcsec = [0.0, 12.0]
science_y_arcsec = [0.0, 12.0]
ngs_x_arcsec = [12.0, -24.0]
ngs_y_arcsec = [-24.0, 12.0]
ngs_magnitude = [15.0, 17.0]
wavelength_um = 2.179
zenith_angle_deg = 20.0
diagnostics_level = validation
```

All coordinates are Cartesian arcseconds; NGS magnitudes follow the fitted
photometry convention. Science x/y arrays must match each other; NGS
x/y/magnitude arrays must match each other. Science and NGS counts are independent.
`output`, all coordinate arrays, magnitudes, wavelength and ZA are required.
`diagnostics_level` defaults to `none`. `non_psd_policy` defaults to `error`
and does not accept `clip` on this path. MASTSEL settings, external flux and
zeropoint inputs are not used. Unsupported `[hybrid]` fields raise.

E2E precision belongs in `[e2e]`, with float32 as the default. Python `dtype`
and CLI `--precision` override it. Fitted products supply their own controller
and photometry settings; calibration controls in the INI do not change those
settings at prediction time. Native sections can remain in the shared INI but
are not imported or executed. Complete prediction opens `science_psf`, the four
model products and `ngs_metrics`; it does not open `ngs_psf` or native sources.
The requested wavelength, ZA, directions, magnitudes and feedback strengths
must lie within the applicable provider/product supports.

The same Python and CLI commands run either selected model:

```bash
hybrid-ao-psf path/to/e2e.ini --precision float64
```
