# Development

This document is the source of truth for the local environment, bootstrap,
Git hook activation, and routine contributor workflow.

## Bootstrap

Until AO Stats has an index release, run bootstrap with its public repository
as the explicit dependency source:

```bash
HYBRID_AO_PSF_AO_STATS_SOURCE='git+https://github.com/nvnunes/ao-stats.git@main' \
  ./scripts/bootstrap.sh
```

The script creates `./.conda`, installs the package with development and
documentation dependencies, configures the versioned Git hook, runs the
canonical checks, and builds the package.

The source override also accepts a local AO Stats checkout for co-development:

```bash
HYBRID_AO_PSF_AO_STATS_SOURCE=../ao-stats ./scripts/bootstrap.sh
```

The override affects only dependency installation. Remote sources are
installed normally; local paths are installed editably. Package metadata
continues to declare the released `ao-stats` distribution contract.

Once AO Stats is available from the configured package index, the ordinary
bootstrap command needs no override:

```bash
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
ownership, artifact schemas, or result persistence.

## Git Hook

The versioned `.githooks/pre-commit` hook runs tests, Ruff, and the strict
documentation build. Bootstrap configures `core.hooksPath` automatically.
