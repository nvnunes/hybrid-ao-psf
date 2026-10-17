# API Guide

This guide defines the supported public Python surface of Hybrid AO PSF.

## Requests And Providers

`HybridRequest` is the single simulation input. Its constructor is keyword-only,
with the NGS flux, magnitude, and zeropoint fields grouped after `ngs_y`.
Cartesian science and NGS coordinates, physical scalars, and NGS inputs are
Astropy quantities.
Array-valued inputs are copied and exposed read-only. `lo_model` accepts
`LowOrderModel.MASTSEL` (the default) or `LowOrderModel.E2E_CALIBRATED`, also
available as `"mastsel"` and `"e2e-calibrated"`.

MASTSEL requests require INI text plus the selected non-PSD and diagnostics
policies. The engine reads `RTC.SensorFrameRate_LO` from that INI for every active NGS.
Supply exactly one of `ngs_flux` (effective `photon / s`) or
`ngs_magnitude` (one `mag` value per active NGS). Magnitudes require a
positive, scalar `ngs_magnitude_zeropoint` in `photon / (m2 s)` supplied by
the caller, not stored in the MASTSEL INI. The engine uses
`telescope.TelescopeDiameter`, the first `sensor_LO.NumberLenslets` value,
and `RTC.SensorFrameRate_LO` with NGS Photometry to resolve effective flux
once before MASTSEL.

E2E requests require magnitudes and use the calibrated model's photometry.
They reject MASTSEL INI text, flux, an external magnitude zeropoint and the
`clip` policy. Product selection remains the caller's responsibility.
Coordinates, Ctot axes, and PSF image orientation follow the
[coordinate contract](architecture.md#coordinate-and-image-orientation).

`SciencePsfProvider` and `NgsMetricProvider` define the two reusable loaded
provider boundaries. Their predictions use `ScienceHoPsfPrediction` and
`NgsHoMetricPrediction`. The science provider owns the PSF pixel scale and
metadata; simulation preserves both and has no independent output-scale
setting. `ScienceHoPsfInterpolator` and
`NgsHoMetricInterpolator` are the concrete immutable providers. Their complete
sample, configuration, build, validation, evaluation, replay, save, and load
surface is exported from both `hybrid_ao_psf` and
`hybrid_ao_psf.interpolation`; see [Interpolation products](interpolation.md).

## E2E Calibration

The package root exports `E2ECalibration`, physical sample/settings/result types,
family-specific measure/build/save/load operations and fitted-product identities.
The four model types are `ReconstructorNoiseModel`, `CovarianceFunctions`,
`ComaResponseModel` and `SensingCalibration`. The configured subclass supplies
native observations; standalone builders, numerical replay and loaded model
evaluation need no native software. See [Calibration](calibration.md) for
stages, physical conventions, configuration and product contracts.

## Calibrated Low-Order Model

`E2ECalibratedModel` owns prepared calibrated prediction independently of
`E2ECalibration`. `load_e2e_calibrated_model()` loads its complete shared INI
inputs; `model.predict()` returns an immutable `E2ELowOrderPrediction` containing
component covariances, feedback diagnostics and displacement-mixture parameters.
This operation calculates LO motion only. See [Calibrated LO prediction](e2e-model.md)
for loading, support, units and the boundary with complete PSF simulation.

## Results

`HybridResult` is the immutable standard result. It carries the final
selected-precision PSF cube, neutral `ao_stats.PsfMetadata`, source metadata, NGS
metrics, angular Ctot, frame rates, jitter, PSD-clipping mask, provenance, model
identity and the applied `LowOrderDistribution`. Its weights and covariances
have shapes `(component,)` and `(science, component, 2, 2)`; their weighted
covariance equals angular Ctot. Execution records field/NGS positions and ZA.
MASTSEL additionally retains wavefront Ctot, angular conversion, effective flux
and INI, and optional supplied magnitudes/zeropoint. Those MASTSEL-only fields
are `None` for E2E, which instead retains magnitudes and a complete
`E2ELowOrderPrediction` with component diagnostics.
`HybridDiagnostics` carries the selected validation summaries. At the debug
level it also carries the large pre-blur PSF cube.

Result arrays and quantities are result-owned and read-only. Final-PSF
statistics are deliberately absent; callers may apply AO Stats when wanted.

Use `save_result()` and `load_result()` for the versioned standalone HDF5
contract described in [Result files](results.md).

## Simulation

`simulate(request, science_provider, ngs_provider=None, *, ctot_correction=None,
e2e_model=None)` is the one stateless scientific operation. MASTSEL requires
`ngs_provider`; E2E requires `e2e_model` and rejects an external NGS provider or
correction callback. The engine evaluates the selected LO prediction, applies
the corresponding Gaussian or full mixture blur, derives jitter and returns
`HybridResult`. Providers are borrowed and
may be reused across calls. See [Engine](engine.md) for the Ctot and diagnostic
policies.

### Covariance Corrections

On the MASTSEL path, supply a `HybridCtotCorrection` to replace the field-wide baseline covariance
before conditioning and the existing single Gaussian blur. The callable receives
`(Nscience, 2, 2)` unconditioned Ctot in `nm2` and a `HybridCtotContext`.
Return a new Quantity convertible to `nm2` with the same shape/order, or `None`
for no change. Do not mutate inputs. Exceptions propagate; the engine still owns
PSD conditioning, convolution and all result diagnostics.

The context borrows the live MASTSEL runtime until the callback returns. It
contains the resolved request, interpolated NGS metrics, effective flux and
frame rates, effective INI, and angular-to-wavefront scale. Its
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
    load_configuration,
    load_ngs_ho_metric_interpolator,
    load_science_ho_psf_interpolator,
    simulate,
)

configuration = load_configuration("examples/hybrid.ini")
correction = HybridCtotCorrection(
    apply=lambda ctot, context: 2 * ctot,
    provenance={"ctot_scale": "2"},
)
result = simulate(
    configuration.request,
    load_science_ho_psf_interpolator(configuration.science_psf_artifact),
    load_ngs_ho_metric_interpolator(configuration.ngs_metric_artifact),
    ctot_correction=correction,
)
```

The definition copies/freezes its string-to-string provenance. The engine
merges it into result provenance even when the callback returns `None`, and
rejects collisions with engine-owned keys. `save_result()` and `load_result()`
preserve this metadata, not the callable or runtime. No correction is enabled
by default, and the INI/CLI does not select one.

`blur_psfs_with_ctot(psfs, ctot_wavefront, *, pixel_scale,
angle_to_wavefront_scale)` is a narrower public operation for externally
supplied PSFs. It accepts a symmetric, already-conditioned wavefront Ctot cube
in `nm2`, checks for materially non-PSD matrices, and returns a separate array
with the same dtype. It uses the engine's finite-field blur without
renormalizing spill, running MASTSEL, or applying a clipping policy. Zero Ctot
leaves the corresponding PSF unchanged.

For comparisons that already have pixel-space covariance or OTFs,
`image_plane_gaussian_otf(covariance_pix2, shape, *, mean_shift_pix=None)`
constructs one Gaussian OTF, and `apply_finite_fov_otfs(psfs, otfs)` applies a
matching OTF cube to PSFs in place. OTF shapes are twice the retained PSF
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

`--output` overrides the path in `[hybrid]`; `--overwrite` explicitly permits
replacement. The command loads the same providers, calls the same `simulate()`
operation, and uses the same `save_result()` contract as Python callers.

Package version inspection is available:

```python
import hybrid_ao_psf

print(hybrid_ao_psf.__version__)
```

See the [generated reference](reference/api.md) for the current root API.
