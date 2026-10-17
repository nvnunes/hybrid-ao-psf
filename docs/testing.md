# Testing

This document defines canonical verification for Hybrid AO PSF source,
dependencies, documentation, and packaging.

## Environment

Use the repository-local environment created by
[Development](development.md). Refresh the editable installation when package
metadata or dependencies change:

```bash
./.conda/bin/python -m pip install -e ".[dev,docs]"
```

## Canonical Commands

Run:

```bash
./.conda/bin/python -m pytest -q
./.conda/bin/ruff check src tests examples
./.conda/bin/mkdocs build --strict
./.conda/bin/python -m build
```

## Completion Expectations

Package-root imports, generated reference, and documented public API must
agree. Automated documentation checks verify navigation targets, local links,
the canonical generated-reference boundary, and the absence of internal
project-history language from public docs. The source distribution must
include the documentation, runnable sample, and packaged regression data
named by the public surfaces. Scientific or interpolation changes require
focused comparison with the project-owned regression baseline.
Persisted-contract changes require round-trip and version-validation tests.
MASTSEL compatibility changes require real execution against the minimum and
current selected releases.

Targeted checks are appropriate during iteration. Complete substantial work
with every canonical command plus the regression checks for the affected
surface.

## PSF Precision Coverage

`test_psf_precision.py` checks explicit float64 source-node pixels, flux,
off-node interpolation and replay through save/load and prediction, while
preserving existing float32 defaults and rejecting unsupported precisions.
It also checks physical metadata, NGS metrics, immutable runtime preparation
and retention of producer precision. `test_execution_precision.py` adds independent FFT comparisons, a source
pixel perturbation below float32 resolution, metrics, both-mode result
round trips, legacy mixed-array preservation and declared-precision rejection.

## Calibration Input Coverage

`tests/test_calibration_boundary.py` uses independent synthetic records for
physical units, pooled-mean input preservation, immutable values/unit metadata,
configuration precedence, schema rejection, atomic sample round trips and
manifest/payload content binding. Native conversion tests belong downstream;
common tests require no native E2E files or software.

## Calibration Pipeline Coverage

`test_calibration_kernels.py` and `test_controller_noise_contracts.py` compare
physical controller/filtering, reconstruction and blur operations with
independent recurrences, impulse sums and closed forms. The four family suites
cover retained observations, offline building, support/units/settings rejection,
immutable model state and numerical persistence. `test_calibration_workflows.py`
exercises all four configured workflows with a synthetic subclass, including
configured spot loading, calibrated reconstruction validation, noise-only replay
and complete shape fitting without native files. Publication tests interrupt
payload writes and manifest replacement and verify complete usable products and
preserved permissions. Native conversion and source/reuse tests belong downstream.
