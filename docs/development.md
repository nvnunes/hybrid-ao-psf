# Development

This document is the source of truth for the local environment, bootstrap,
Git hook activation, and routine contributor workflow.

## Bootstrap

Run bootstrap with the public AO Stats repository as the explicit dependency
source:

```bash
HYBRID_AO_PSF_AO_STATS_SOURCE='git+https://github.com/nvnunes/ao-stats.git@main' \
  ./scripts/bootstrap.sh
```

The script creates `./.conda`, installs the package with development and
documentation dependencies, configures the versioned Git hook, runs the
canonical checks, and builds the package.

The AO Stats source override also accepts a local checkout for co-development:

```bash
HYBRID_AO_PSF_AO_STATS_SOURCE=../ao-stats ./scripts/bootstrap.sh
```

Bootstrap also installs NGS Photometry from its public GitHub repository
before installing Hybrid AO PSF. For a sibling checkout, set
`HYBRID_AO_PSF_NGS_PHOTOMETRY_SOURCE=../ngs-photometry`; remote sources are
installed normally and local paths editably. These overrides affect only
dependency installation. Package metadata sets the required distribution
versions.

## AO Models

Hybrid requires AO Models 0.2.0 or later for shared configuration, control
and reconstruction. Obtain the AO Models checkout or a wheel from the
project maintainer. In the joint
development workspace, bootstrap uses the sibling `../ao-models` checkout.

Set `HYBRID_AO_PSF_AO_MODELS_SOURCE` to use a different local checkout, a
wheel file, or a VCS URL. For example, to use a wheel supplied separately:

```bash
HYBRID_AO_PSF_AO_STATS_SOURCE='git+https://github.com/nvnunes/ao-stats.git@main' \
HYBRID_AO_PSF_AO_MODELS_SOURCE=/path/to/ao_models-0.2.0-py3-none-any.whl \
  ./scripts/bootstrap.sh
```

## Daily Commands

Refresh the editable installation after dependency or metadata changes:

```bash
./.conda/bin/python -m pip install -e ".[dev,docs]"
```

Serve the documentation locally with:

```bash
./.conda/bin/mkdocs serve
```

Use [Testing](testing.md) for canonical verification. Consult
[Architecture](architecture.md) before changing public contracts, scientific
ownership, file schemas, or result persistence.

## Git Hook

The versioned `.githooks/pre-commit` hook runs tests, Ruff, and the strict
documentation build. Bootstrap configures `core.hooksPath` automatically.
