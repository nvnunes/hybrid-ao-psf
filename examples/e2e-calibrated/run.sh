#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"
bin=../../.conda/bin

"$bin/python" ../interpolation/build_interpolators.py \
  --science-psfs ../interpolation/samples/science-psfs \
  --ngs-psfs ../interpolation/samples/ngs-psfs \
  --skip-ngs-psf \
  --output-dir products

"$bin/python" generate_synthetic_telemetry.py calibrate.ini \
  --ngs-metrics products/ngs-metrics.pkl --output-dir products/telemetry
"$bin/python" calibrate.py calibrate.ini

"$bin/hybrid-ao-psf" hybrid.ini --precision float64 --overwrite
