#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"
bin=../../.conda/bin

"$bin/python" ../interpolation/build_interpolators.py \
  --science-psfs ../interpolation/samples/science-psfs \
  --ngs-psfs ../interpolation/samples/ngs-psfs \
  --skip-ngs-psf \
  --output-dir products

"$bin/hybrid-ao-psf" hybrid.ini --overwrite
