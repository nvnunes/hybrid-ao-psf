# Hybrid AO PSF

Hybrid AO PSF produces adaptive-optics science PSFs from interpolated
high-order PSFs and MASTSEL low-order residuals. It provides a standalone
research interface and a typed Python boundary for larger orchestration
systems.

The package provides immutable request and result contracts, two reusable
interpolation-product families, combined-INI loading, versioned HDF5 result
persistence, a stateless MASTSEL-backed simulation operation, and standalone
configuration-to-file execution through Python or the CLI.

## Start Here

- Use the [API guide](api.md) for the supported package surface.
- Use [Configuration](configuration.md) for the combined Hybrid-and-MASTSEL
  INI contract.
- Use [Interpolation products](interpolation.md) to build, validate, persist,
  load, and evaluate the two provider families.
- Read [Engine](engine.md) for the stateless simulation and Ctot policy.
- Use [Result files](results.md) for the standalone HDF5 schema.
- Read [Architecture](architecture.md) for ownership and dependency boundaries.
- Use [Development](development.md) to create the local environment.
- Use [Testing](testing.md) for canonical verification commands.
