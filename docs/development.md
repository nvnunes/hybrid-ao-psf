# Development

This document is the source of truth for the local environment, bootstrap,
Git hook activation, and routine contributor workflow.

## Bootstrap

Run bootstrap from the repository root:

```bash
./scripts/bootstrap.sh
```

The script creates `./.conda`, installs the package with development and
documentation dependencies, configures the versioned Git hook, runs the
canonical checks, and builds the package. It first reinstalls AO Core from
GitHub and verifies the PSF measurements, photometry and configuration imports.

Set `HYBRID_AO_PSF_AO_CORE_SOURCE` to a sibling checkout, wheel or VCS URL.
For co-development, a local checkout is installed editably:

```bash
HYBRID_AO_PSF_AO_CORE_SOURCE=../ao-core ./scripts/bootstrap.sh
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
