# Hybrid AO PSF

Hybrid AO PSF is a standalone Python engine for producing adaptive-optics
science PSFs by combining interpolated high-order PSFs with either MASTSEL or
E2E-calibrated low-order motion. Researchers can run one simulation directly
from an INI file or use the typed Python API from a larger orchestration system.

The project owns Hybrid-specific interpolation products, both low-order models,
scientific execution, engine diagnostics, and standalone results. AO Core
provides shared configuration, control, reconstruction, photometry, and
PSF metadata and measurements.

## Status

The package is pre-alpha. Its immutable request and result contracts,
combined-INI loader, versioned interpolation and result persistence, stateless
`simulate()` engine, and INI/CLI route pass the project-owned scientific
regression suite on the minimum and current selected MASTSEL releases.

## Installation

Install an AO Core build with `ao_core.models` and `ao_core.stats`.
This example uses a sibling AO Core checkout.

```bash
python -m pip install --force-reinstall --no-deps ../ao-core
python -m pip install .
python -c 'import ao_core.photometry, ao_core.stats; from ao_core.models.config import Configuration'
```

For repository-local development and source overrides, see
[Development](docs/development.md).

## Python Quick Start

Run a selected-model Hybrid INI and receive the validated immutable
result:

```python
from hybrid_ao_psf import run_configuration

result = run_configuration("path/to/simulation.ini")
print(result.psfs.shape)
```

The configured HDF5 result is written at the same time. For callers that
already manage loaded interpolation providers, `simulate()` is the lower-level
stateless operation. MASTSEL is the default; see
[Calibrated prediction](docs/e2e-model.md) for the calibrated model and
[Configuration](docs/configuration.md) for both INI formats.

## Development Setup

Bootstrap the repository-local environment:

```bash
./scripts/bootstrap.sh
```

The bootstrap installs development and documentation dependencies, activates
the versioned Git hook, runs the tests and Ruff, builds the documentation in
strict mode, and builds the package.

## Standalone Examples

Generate synthetic J-, H-, and K-band science PSFs and single-wavelength
NGS PSFs, then run the E2E-Calibrated example from the repository root.

```bash
./.conda/bin/python examples/interpolation/generate_synthetic_ho_psfs.py \
  --output-dir examples/interpolation/samples
./examples/e2e-calibrated/run.sh
```

The script builds the interpolators and synthetic calibration products in
`examples/e2e-calibrated/products/` and writes the result to
`examples/e2e-calibrated/results/hybrid.h5`. To use your own PSFs, set the
builder's input paths in the run script to directories in the
[example PSF library format](docs/interpolation.md#example-psf-libraries).
The example demonstrates prediction with synthetic calibration products,
even when you supply your own PSFs. See [E2E Calibration](docs/calibration.md)
for the measurements that produce models for a physical system.

See [Configuration](docs/configuration.md#executable-sample) for the MASTSEL example
and [Calibrated Low-Order Prediction](docs/e2e-model.md) for the Python entry
points.

## Documentation

- [API guide](docs/api.md)
- [Configuration](docs/configuration.md)
- [Interpolation products](docs/interpolation.md)
- [Engine](docs/engine.md)
- [Result files](docs/results.md)
- [Architecture](docs/architecture.md)
- [Development](docs/development.md)
- [Testing](docs/testing.md)
- [Generated Python reference](docs/reference/api.md)

## License

Hybrid AO PSF is distributed under the MIT License. See [LICENSE](LICENSE).
