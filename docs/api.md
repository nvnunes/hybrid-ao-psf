# API Guide

This guide defines the supported public Python surface of Hybrid AO PSF.

## Requests And Providers

`HybridRequest` describes the observation and requested diagnostics. Its
keyword-only inputs use Astropy quantities. Supply `science_positions` and
`ngs_positions` as angular arrays of shape `(N, 2)`, with an `(x, y)` pair per
row. Supply the WFS-effective photon rate as `ngs_flux` in `photon / s`, or
supply `ngs_magnitude` in `mag`. Give one value per NGS in the same order.
Array inputs are read-only copies.

Pass a `MastselModel` or `E2ECalibratedModel` as `lo_model` to `simulate()`.
`MastselModel` holds the INI text, NGS metric provider, non-PSD policy and
optional covariance correction. For magnitudes, set the model's positive scalar
`ngs_magnitude_zeropoint` in `photon / (m2 s)`. The engine uses
`telescope.TelescopeDiameter`, the first `sensor_LO.NumberLenslets` value,
and `RTC.SensorFrameRate_LO` with `ao_core.photometry` to calculate
WFS-effective photon rates once before calling MASTSEL.

E2E-Calibrated takes magnitudes and supplies its own NGS metrics and photometry.
Coordinates, Ctot axes, and PSF image orientation follow the
[coordinate contract](architecture.md#coordinate-and-image-orientation).

`simulate()` accepts `science_psfs` implementing `PsfProvider`. A `MastselModel`
accepts `ngs_metrics` implementing `PsfMetricProvider`. These contracts
define the requested coordinates and the returned images or metrics.
The source of the data belongs to the implementation. Both contracts expose
`evaluate()` for requested coordinates. Build or load an interpolator, then
evaluate it directly or pass it to `simulate()`.

`PsfInterpolator` supplies interpolated images. `PsfFileProvider` supplies
existing images through a caller-supplied reader. Both return `PsfPrediction`
with the images' pixel scale and physical metadata.
`PsfMetricInterpolator` returns `PsfMetricPrediction`.
See [Interpolation products](interpolation.md) for building interpolators
and reading saved PSFs.

## E2E Calibration

Use `load_telemetry_calibration()` for saved HDF5 telemetry configured in an
INI. Python callers can also construct `TelemetryCalibration` with file paths
and system settings. Subclass `E2ECalibration` to supply live native
measurements. Both workflows build from reusable measurement records and
save explicitly.
See [Calibration](calibration.md#calibration-from-hdf5) for the file-backed
workflow, physical conventions and model contracts.

## Calibrated Low-Order Model

`E2ECalibratedModel` owns prepared calibrated prediction independently of
`E2ECalibration`. `load_e2e_calibrated_model()` loads its complete shared INI
inputs. `model.predict()` returns an immutable `E2ELowOrderPrediction` with
a `displacement_distribution` at the requested science positions. Pass
`diagnostics_level="debug"` to retain the evaluated NGS inputs, control
solution and intermediate covariance terms. See [Calibrated LO prediction](e2e-model.md).

## Results

`HybridResult` returns the ordered Cartesian science `positions`, final
science `psfs` and radial `jitter`. Row `i` identifies the same field point
in all three. Its physical `metadata` is an `ao_core.stats.PsfMetadata` record
accepted by the `ao_core.stats` measurement functions. The arrays are
read-only and retain the simulation precision.

Optional `diagnostics` provide validation and debug information
from the low-order calculation. See [Engine diagnostics](engine.md#diagnostics)
for the data provided at each level. Use `diagnostic_fields()` and
`diagnostic_values()` to save the same diagnostic fields through AO Core from
a batch workflow.

Use `run_configuration()` to run and save an INI observation with its inputs.
Use `save_result()` to save the science output and diagnostics of a
`HybridResult`, and `load_result()` to read them back. See [Standalone HDF5 results](results.md).

## Simulation

`simulate(request, science_psfs, lo_model=model)` runs one observation.
The engine evaluates the selected LO prediction, applies
the corresponding Gaussian or full mixture blur, derives jitter and returns
`HybridResult`. Providers are borrowed and
may be reused across calls. See [Engine](engine.md) for the Ctot and diagnostic
policies.

### Covariance Corrections

Set `MastselModel.ctot_correction` to a `HybridCtotCorrection` to replace the
field-wide baseline covariance before conditioning and Gaussian blur. The callable receives
`(Nscience, 2, 2)` unconditioned Ctot in `nm2` and a `HybridCtotContext`.
Return a new Quantity convertible to `nm2` with the same shape/order, or `None`
for no change. Do not mutate inputs. Exceptions propagate; the engine still owns
PSD conditioning, convolution and all result diagnostics.

The context contains the observation request, NGS metrics, angular-to-wavefront
scale, LO frame rate, and the zeropoint used for magnitude inputs. It supplies
the configured numeric loop gain and delay. Automatic gain is reported as
`None`. The required statistics give read-only Css, Cas, centre Cas,
science-position reconstructor blocks, and filtered NGS measurement-noise
covariance for fixed gain. Covariances are in `nm2`, axes are interleaved NGS
`(x, y)`, and reconstructors are dimensionless.

This example doubles Ctot using the providers and request from a combined INI.
First generate the PSFs and build the interpolators for the
[MASTSEL example](configuration.md#executable-sample).

```python
from hybrid_ao_psf import (
    HybridCtotCorrection,
    MastselModel,
    load_configuration,
    load_psf_metric_interpolator,
    load_psf_interpolator,
    simulate,
)

configuration = load_configuration("examples/mastsel/hybrid.ini")
correction = HybridCtotCorrection(apply=lambda ctot, context: 2 * ctot)
model = MastselModel(
    mastsel_ini=configuration.mastsel_ini,
    ngs_metrics=load_psf_metric_interpolator(configuration.ngs_metric_file),
    non_psd_policy=configuration.non_psd_policy,
    ctot_correction=correction,
)
science = load_psf_interpolator(configuration.science_psf_file)
result = simulate(configuration.request, science, lo_model=model)
```

A correction can keep its input paths in the `HybridCtotCorrection.input_files`
mapping. The correction is configured through the Python API.

Use `blur_psfs_with_ctot()` for externally supplied PSFs with symmetric,
already-conditioned wavefront Ctot in square nanometres. Supply the angular
pixel scale and the angle-to-wavefront conversion. The result uses the
selected `dtype`, which defaults to float32. Pass `dtype=np.float64` for
float64 calculations and results. The finite-field blur retains the flux
inside the crop and rejects materially non-PSD covariance.

`image_plane_gaussian_otf()` takes pixel-space covariance and constructs a
Gaussian OTF at the requested precision (float32 by default).
`apply_finite_fov_otfs()` applies a matching OTF cube to PSFs in place.
OTF shapes are twice the retained PSF
height and width; both operations use NumPy's unshifted FFT convention. The
application crops and clips numerical negatives without renormalizing flux.

## Configuration

`load_configuration()` reads one model-specific INI and returns
`HybridConfiguration`. It resolves paths relative to the INI. MASTSEL uses
familiar polar source fields; E2E uses Cartesian `[hybrid]` inputs and the shared
`[e2e]` product/provider paths. See
[Configuration](configuration.md) for the field contract.

`simulate_configuration()` loads the selected model and providers and
returns the in-memory result. `run_configuration()` also writes the HDF5
file and is the operation used by the CLI.

## Standalone Execution

Run one INI directly:

```bash
hybrid-ao-psf path/to/simulation.ini
```

`--output` overrides the path in `[hybrid]`. An explicit relative path is
relative to the working directory. Paths in the INI are relative to the INI.
Pass `--overwrite` to replace an existing result.
The command calls `run_configuration()`, which captures the extra file
fields during the same calculation that produces the in-memory result.

Package version inspection is available:

```python
import hybrid_ao_psf

print(hybrid_ao_psf.__version__)
```

See the [generated reference](reference/api.md) for the current root API.
