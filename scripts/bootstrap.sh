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
if [ -n "${HYBRID_AO_PSF_AO_STATS_SOURCE:-}" ]; then
  case "$HYBRID_AO_PSF_AO_STATS_SOURCE" in
    git+*|http://*|https://*)
      conda run --prefix "$env_prefix" python -m pip install "$HYBRID_AO_PSF_AO_STATS_SOURCE"
      ;;
    *)
      conda run --prefix "$env_prefix" python -m pip install -e "$HYBRID_AO_PSF_AO_STATS_SOURCE"
      ;;
  esac
fi
ngs_photometry_source="${HYBRID_AO_PSF_NGS_PHOTOMETRY_SOURCE:-git+https://github.com/nvnunes/ngs-photometry.git}"
case "$ngs_photometry_source" in
  git+*|http://*|https://*)
    conda run --prefix "$env_prefix" python -m pip install "$ngs_photometry_source"
    ;;
  *)
    conda run --prefix "$env_prefix" python -m pip install -e "$ngs_photometry_source"
    ;;
esac
conda run --prefix "$env_prefix" python -m pip install -e ".[dev,docs]"

echo "[bootstrap] Configuring Git hooks path..."
git config core.hooksPath .githooks

echo "[bootstrap] Running checks..."
conda run --prefix "$env_prefix" python -m pytest -q
conda run --prefix "$env_prefix" ruff check src tests examples
conda run --prefix "$env_prefix" mkdocs build --strict
conda run --prefix "$env_prefix" python -m build

echo "[bootstrap] Done."
