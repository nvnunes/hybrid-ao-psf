# API Guide

This guide defines the supported public Python surface of Hybrid AO PSF.

## Requests And Providers

`HybridRequest` is the single resolved simulation input. Cartesian science and
NGS coordinates, physical scalars, NGS fluxes, and NGS frame rates are Astropy
quantities. Array-valued inputs are copied and exposed read-only. The request
also contains the MASTSEL INI text plus the selected non-PSD and diagnostics
policies.

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
