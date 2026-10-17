# Configuration

Hybrid AO PSF uses one INI file for one simulation. Existing MASTSEL sections
retain their familiar names and fields. The separate `[hybrid]` section owns
artifact selection, standalone output, resolved high-order inputs, and engine
policy.

## Hybrid Section

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

## MASTSEL-Owned Inputs

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

## Example Shape

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

## Executable Sample

The repository's `examples/hybrid.ini` contains a complete small MASTSEL
configuration. From a bootstrapped checkout:

```bash
./.conda/bin/python examples/build_sample_artifacts.py
./.conda/bin/hybrid-ao-psf examples/hybrid.ini
```

The builder creates the two ready-to-evaluate products referenced by the INI.
The command writes the configured `examples/results/hybrid.h5` result. The
sample is also the Python-and-CLI real-MASTSEL end-to-end verification route.

### Calibration Precision

The optional `[e2e] precision = float32|float64` selects shared calculation
precision (default float32). `E2ECalibrationConfiguration.dtype` is the resolved
NumPy dtype. `with_overrides(dtype=...)` overrides the INI; omitting it retains
the INI choice. Native WFS arithmetic remains owned by the native adapter.
