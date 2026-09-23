# Hybrid AO PSF

Hybrid AO PSF is a standalone Python engine for producing adaptive-optics
science PSFs by combining interpolated high-order PSFs with MASTSEL low-order
residuals. Researchers can run one simulation directly from a combined INI
file or use the typed Python API from a larger orchestration system.

The project owns Hybrid-specific interpolation products, MASTSEL-backed
scientific execution, engine diagnostics, and standalone results. It depends
on [AO Stats](https://github.com/nvnunes/ao-stats) for neutral PSF metadata and
measurements, and [NGS Photometry](https://github.com/nvnunes/ngs-photometry)
for magnitude-to-effective-flux conversion.

## Status

The package is pre-alpha. Its immutable request and result contracts,
combined-INI loader, versioned interpolation and result persistence, stateless
`simulate()` engine, and INI/CLI route pass the project-owned scientific
regression suite on the minimum and current selected MASTSEL releases.

## Installation

From a source checkout, install AO Stats and NGS Photometry from their public
repositories, then install Hybrid AO PSF:

```bash
python -m pip install \
  'ao-stats @ git+https://github.com/nvnunes/ao-stats.git@main'
python -m pip install \
  'ngs-photometry @ git+https://github.com/nvnunes/ngs-photometry.git'
python -m pip install .
```

For repository-local development and source overrides, see
[Development](docs/development.md).

## Python Quick Start

Run a combined Hybrid-and-MASTSEL INI and receive the validated immutable
result:

```python
from hybrid_ao_psf import run_configuration

result = run_configuration("path/to/simulation.ini")
print(result.psfs.shape)
```

The configured HDF5 result is written at the same time. For callers that
already manage loaded interpolation providers, `simulate()` is the lower-level
stateless operation.

## Development Setup

Bootstrap the repository-local environment with the current AO Stats source:

```bash
HYBRID_AO_PSF_AO_STATS_SOURCE='git+https://github.com/nvnunes/ao-stats.git@main' \
  ./scripts/bootstrap.sh
```

The bootstrap installs development and documentation dependencies, activates
the versioned Git hook, runs the tests and Ruff, builds the documentation in
strict mode, and builds the package.

## Standalone Sample

Build the small example interpolation products and run one simulation:

```bash
./.conda/bin/python examples/build_sample_artifacts.py
./.conda/bin/hybrid-ao-psf examples/hybrid.ini
```

The combined INI resolves its artifact and result paths relative to
`examples/hybrid.ini` and writes `examples/results/hybrid.h5`. The same
scientific operation is available programmatically through `simulate()`;
`run_configuration()` provides the complete INI-to-HDF5 convenience route.

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
