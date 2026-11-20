#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

if ! command -v conda >/dev/null 2>&1; then
  echo "Error: conda is required but not found in PATH."
  exit 1
fi

env_prefix="$repo_root/.conda"

if [ ! -d "$env_prefix" ]; then
  echo "[bootstrap] Creating conda environment at $env_prefix..."
  conda create -y --prefix "$env_prefix" python=3.12 pip
fi

echo "[bootstrap] Installing package with dev+docs extras..."
ao_core_source="${HYBRID_AO_PSF_AO_CORE_SOURCE:-git+https://github.com/nvnunes/ao-core.git}"
case "$ao_core_source" in
  git+*|http://*|https://*|*.whl)
    conda run --prefix "$env_prefix" python -m pip install --force-reinstall --no-deps "$ao_core_source"
    ;;
  *)
    conda run --prefix "$env_prefix" python -m pip install --force-reinstall --no-deps -e "$ao_core_source"
    ;;
esac
conda run --prefix "$env_prefix" python -m pip install -e ".[dev,docs]"
conda run --prefix "$env_prefix" python -c 'import ao_core.photometry, ao_core.stats
from ao_core.models.config import Configuration
from ao_core.models.control import TipTiltControl
from ao_core.models.residual import assemble_tilt_residual_covariances
import hybrid_ao_psf'

echo "[bootstrap] Configuring Git hooks path..."
git config core.hooksPath .githooks

echo "[bootstrap] Running checks..."
conda run --prefix "$env_prefix" python -m pytest -q
conda run --prefix "$env_prefix" ruff check src tests examples
conda run --prefix "$env_prefix" mkdocs build --strict
conda run --prefix "$env_prefix" python -m build

echo "[bootstrap] Done."
