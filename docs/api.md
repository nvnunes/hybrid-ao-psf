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

The package root exports `E2ECalibration`, physical sample/settings/result types,
family-specific measure/build/save/load operations and calibrated model file checks.
The four model types are `ReconstructorNoiseModel`, `CovarianceFunctions`,
`ComaResponseModel` and `SensingModel`. The configured subclass supplies
native measurements; standalone builders, numerical simulation and loaded model
evaluation need no native software. See [Calibration](calibration.md) for
stages, physical conventions, configuration and product contracts.

## Calibrated Low-Order Model

`E2ECalibratedModel` owns prepared calibrated prediction independently of
`E2ECalibration`. `load_e2e_calibrated_model()` loads its complete shared INI
inputs; `model.predict()` returns an immutable `E2ELowOrderPrediction` containing
component covariances, feedback diagnostics and a `LowOrderDistribution` at
`prediction.distribution`. This operation calculates LO motion only. See [Calibrated LO prediction](e2e-model.md)
for loading, support, units and the boundary with complete PSF simulation.

## Results

`HybridResult` is the immutable standard result. It carries the final
selected-precision PSF cube, neutral `ao_core.stats.PsfMetadata`, NGS
metrics, angular Ctot, frame rates, jitter, PSD-clipping mask, provenance, model
identity and the applied `LowOrderDistribution` at `result.distribution`. Its
weights and covariances have shapes `(component,)` and
`(science, component, 2, 2)`. Their weighted covariance equals angular Ctot.
Execution records field/NGS positions and ZA.
MASTSEL additionally retains wavefront Ctot, angular conversion, effective flux
and INI, and optional supplied magnitudes/zeropoint. Those MASTSEL-only fields
are `None` for E2E, which instead retains magnitudes and a complete
`E2ELowOrderPrediction` with component diagnostics.
`HybridDiagnostics` carries the selected validation summaries. At the debug
level it also carries the large pre-blur PSF cube.

Result arrays and quantities are result-owned and read-only. Final-PSF
statistics are deliberately absent; callers may use `ao_core.stats` when wanted.

Use `save_result()` and `load_result()` for the versioned standalone HDF5
contract described in [Result files](results.md).

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

The context borrows the live MASTSEL runtime until the callback returns. It
contains the observation request, interpolated NGS metrics, WFS-effective photon
rates, frame rates, the zeropoint used for magnitude inputs, the effective INI
and the angular-to-wavefront scale. Its
`HybridMastselStatistics` contains read-only Css, Cas, centre Cas, exact
science-position reconstructor blocks and filtered NGS measurement-noise
covariance. Covariances are in `nm2`, axes are interleaved NGS `(x, y)`, and
reconstructors are dimensionless. Noise is available for fixed controller gain;
otherwise it is `None`. These snapshots and the live runtime are not persisted.
Do not retain the context or re-evaluate baseline MASTSEL from the callback.

This example doubles Ctot using the providers and request from a combined INI.
First build the sample products from the project root:

```bash
./.conda/bin/python examples/build_sample_artifacts.py
```

```python
from hybrid_ao_psf import (
    HybridCtotCorrection,
    MastselModel,
    load_configuration,
    load_psf_metric_interpolator,
    load_psf_interpolator,
    simulate,
)

configuration = load_configuration("examples/hybrid.ini")
correction = HybridCtotCorrection(
    apply=lambda ctot, context: 2 * ctot,
    provenance={"ctot_scale": "2"},
)
model = MastselModel(
    mastsel_ini=configuration.mastsel_ini,
    ngs_metrics=load_psf_metric_interpolator(configuration.ngs_metric_file),
    non_psd_policy=configuration.non_psd_policy,
    ctot_correction=correction,
)
science = load_psf_interpolator(configuration.science_psf_file)
result = simulate(configuration.request, science, lo_model=model)
```

The definition copies/freezes its string-to-string provenance. The engine
merges it into result provenance even when the callback returns `None`, and
rejects collisions with engine-owned keys. `save_result()` and `load_result()`
preserve this metadata, not the callable or runtime. No correction is enabled
by default, and the INI/CLI does not select one.

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

`simulate_configuration()` loads the selected model/providers and calls
`simulate()`. `run_configuration()` additionally writes the configured HDF5
result and is the operation used by the CLI.

## Standalone Execution

Run one INI directly:

```bash
hybrid-ao-psf path/to/simulation.ini
```

`--output` overrides the path in `[hybrid]`. An explicit relative path is
relative to the working directory. Paths in the INI are relative to the INI.
Pass `--overwrite` to replace an existing result.
The command loads the same providers, calls the same `simulate()`
operation, and uses the same `save_result()` contract as Python callers.

Package version inspection is available:

```python
import hybrid_ao_psf

print(hybrid_ao_psf.__version__)
```

See the [generated reference](reference/api.md) for the current root API.
