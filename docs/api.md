# API Guide

This guide defines the supported public Python surface of Hybrid AO PSF.

## Requests And Providers

`HybridRequest` is the single resolved simulation input. Cartesian science and
NGS coordinates, physical scalars, and NGS fluxes are Astropy quantities.
Array-valued inputs are copied and exposed read-only. The request also contains
the MASTSEL INI text plus the selected non-PSD and diagnostics policies. The
engine reads `RTC.SensorFrameRate_LO` from that INI for every active NGS.
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

## Results

`HybridResult` is the immutable standard result. It carries the final
`float32` PSF cube, neutral `ao_stats.PsfMetadata`, source metadata, NGS
metrics, Ctot in wavefront and angular units, conversion scale, NGS inputs,
jitter, effective MASTSEL configuration, PSD-clipping mask, and provenance.
`HybridDiagnostics` carries the selected validation summaries. At the debug
level it also carries the large pre-blur PSF cube.

Result arrays and quantities are result-owned and read-only. Final-PSF
statistics are deliberately absent; callers may apply AO Stats when wanted.

Use `save_result()` and `load_result()` for the versioned standalone HDF5
contract described in [Result files](results.md).

## Simulation

`simulate(request, science_provider, ngs_provider)` is the one stateless
scientific operation. It evaluates both loaded providers, creates the effective
MASTSEL configuration, computes and validates Ctot, applies the finite-field
blur, derives jitter, and returns `HybridResult`. Providers are borrowed and
may be reused across calls. See [Engine](engine.md) for the Ctot and diagnostic
policies.

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

`load_configuration()` reads one combined Hybrid-and-MASTSEL INI, resolves
paths relative to the INI, converts familiar polar source coordinates into a
Cartesian `HybridRequest`, and returns `HybridConfiguration`. See
[Configuration](configuration.md) for the field contract.

`simulate_configuration()` loads the two configured artifacts and calls
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
