# Configuration

Hybrid AO PSF uses one INI file for one simulation. MASTSEL is the default.
For E2E-Calibrated prediction, set `[hybrid]` `LOModel = e2e-calibrated`.
Paths resolve relative to the INI.
The loader assembles each model's INI coordinates into Cartesian `(x, y)` rows
for `HybridRequest.science_positions` and `HybridRequest.ngs_positions`.

## MASTSEL Configuration

Existing MASTSEL sections retain their familiar names and fields. The separate `[hybrid]` section owns
file selection, standalone output, resolved high-order inputs, and engine
policy.

### Hybrid Section

The `[hybrid]` section requires:

| Field | Meaning |
| --- | --- |
| `science_psf_file` | Loaded science-HO-PSF file path |
| `ngs_metric_file` | Loaded NGS-HO-metric file path |
| `output` | Standalone HDF5 result path |
| `ngs_magnitude` | One magnitude per active NGS |
| `ngs_magnitude_zeropoint` | Pre-aperture zero-magnitude flux in photons per square metre per second |

File and output paths are resolved relative to the INI file. The file
paths must exist when the configuration is loaded; the output and its parent
directory may be created later.

Set `ngs_mag_standard` when the magnitudes have a known photometric convention.
Hybrid converts magnitudes to effective WFS photon rates using the MASTSEL
telescope diameter and lenslet count. A legacy INI with `ngs_flux_photon_per_s`
supplies effective rates directly. The INI loader converts them to equivalent
magnitudes using the reference zeropoint of `1e10 photon / (m2 s)`.

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

The loaded science-PSF file owns the PSF pixel scale. Hybrid AO PSF uses
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
science_psf_file = products/science.pkl
ngs_metric_file = products/ngs-metrics.pkl
output = results/hybrid.h5
ngs_magnitude = [14.0, 15.0, 16.0]
ngs_magnitude_zeropoint = 1e10
non_psd_policy = error
diagnostics_level = none
```

This abbreviated example omits other MASTSEL sections required by real engine
execution. Hybrid AO PSF passes those sections through its owned translation
boundary rather than reclassifying them as Hybrid fields.

### Executable Sample

The repository's `examples/mastsel/hybrid.ini` contains a complete small
MASTSEL configuration. Run these commands from the repository root.

```bash
./.conda/bin/python examples/interpolation/generate_synthetic_ho_psfs.py \
  --output-dir examples/interpolation/samples
./examples/mastsel/run.sh
```

The runner builds `science.pkl` and `ngs-metrics.pkl` in
`examples/mastsel/products/` and writes `examples/mastsel/results/hybrid.h5`.
To use your own PSFs, set the
builder's input paths in the run script to directories in the
[example PSF library format](interpolation.md#example-psf-libraries).
The sample is also the Python-and-CLI real-MASTSEL end-to-end verification route.

## E2E-Calibrated Configuration

Use the model files written by [calibration](calibration.md#shared-configuration).
Set the model and interpolator paths in `[inputs]` and the result path in
`[outputs]`. Use the standard AO sections for observing conditions and
`[hybrid]` for the Cartesian positions.

```ini
[inputs]
SciencePSFPath = products/science.pkl
NGSMetricsPath = products/ngs-metrics.pkl
ReconstructorModelPath = products/reconstructor/model.json
CovarianceModelPath = products/covariance/model.json
ComaModelPath = products/coma/model.json
SensingModelPath = products/sensing/model.json

[outputs]
OutputPath = results/hybrid.h5

[telescope]
ZenithAngle = 20

[sources_science]
Wavelength = [2.179e-6]

[hybrid]
LOModel = e2e-calibrated
ScienceX = [0.0, 12.0]
ScienceY = [0.0, 12.0]
NGSX = [12.0, -24.0]
NGSY = [-24.0, 12.0]
NGSMagnitude = [15.0, 17.0]
DiagnosticsLevel = validation
```

Give Cartesian coordinates in arcseconds, `Wavelength` in metres, and
`ZenithAngle` in degrees. NGS magnitudes follow the fitted
photometry convention. Science x/y arrays must match each other; NGS
x/y/magnitude arrays must match each other. Science and NGS counts are independent.
`OutputPath`, all coordinate arrays, magnitudes, wavelength and ZA are required.
`DiagnosticsLevel` defaults to `none`. `NonPSDPolicy` defaults to `error`
and does not accept `clip` on this path. MASTSEL settings, external flux and
zeropoint inputs are not used. Unsupported `[hybrid]` fields raise.

Fitted products supply their own controller and photometry settings.
Prediction loads the four fitted models, `SciencePSFPath`, and `NGSMetricsPath`.
The requested wavelength, ZA, directions, magnitudes and feedback strengths
must lie within the applicable provider/product supports.

### Executable E2E-Calibrated Example

Generate the sample PSFs and run the example from the repository root.

```bash
./.conda/bin/python examples/interpolation/generate_synthetic_ho_psfs.py \
  --output-dir examples/interpolation/samples
./examples/e2e-calibrated/run.sh
```

The runner works from `examples/e2e-calibrated/`. It builds the PSF
interpolators, then runs the following commands.

```bash
bin=../../.conda/bin
"$bin/python" generate_synthetic_telemetry.py calibrate.ini \
  --ngs-metrics products/ngs-metrics.pkl --output-dir products/telemetry
"$bin/python" calibrate.py calibrate.ini
"$bin/hybrid-ao-psf" hybrid.ini --precision float64 --overwrite
```

`calibrate.ini` lists the telemetry files in `[inputs]` and the four
model destinations in `[outputs]`. `ReconstructorTelemetryPath` names a single file.
`CovarianceTelemetryPath`, `ComaTelemetryPath`, and `SensingTelemetryPath`
are formats relative to the INI. Use `{ZA}` to insert the angle in
degrees without trailing zeros or exponent notation. The adapter uses the AO system settings and
telemetry described in
[HDF5 Telemetry](calibration.md#hdf5-telemetry). The
`[sensing]` keys `OneNGSAsterisms`, `TwoNGSAsterisms`, and
`ThreeNGSAsterisms` set the counts per zenith angle. `TelemetryCalibration`
selects asterisms from the sensing grid for the noise-blur fit. The fit requires single- and multi-NGS
cases. The reader takes the grid axes and measurement counts from the telemetry.
The synthetic generator chooses its sampling before it writes those files.

For the synthetic example, the generator samples spot EE and FWHM from the NGS
metric interpolator and saves them in the sensing telemetry. `calibrate.py` uses
the saved values when running the blur fit.
`hybrid.ini` specifies the prediction and points to the fitted models.
The prediction writes
`results/hybrid.h5`.

To substitute your own captured files, follow
[Calibration From HDF5](calibration.md#calibration-from-hdf5).
