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

Bootstrap reinstalls AO Core from its public GitHub repository before
installing Hybrid AO PSF. Hybrid requires an AO Core build with its
photometry, shared configuration, control and reconstruction modules.
Bootstrap checks the photometry import and the shared `Configuration` API.

Set `HYBRID_AO_PSF_AO_CORE_SOURCE` to use a sibling checkout, wheel, or VCS
URL. Bootstrap installs a local checkout editably. Use the sibling
checkout during co-development:

```bash
HYBRID_AO_PSF_AO_STATS_SOURCE='git+https://github.com/nvnunes/ao-stats.git@main' \
HYBRID_AO_PSF_AO_CORE_SOURCE=../ao-core \
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
