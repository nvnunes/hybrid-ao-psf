"""Versioned interpolation files for Hybrid workflows.

This public submodule owns the Hybrid AO PSF file contract for
PSF image and metric interpolation.
Objects are built, evaluated, compared with reference samples, saved, and
loaded through the functions exported here. Validation is internal to those
operations. The supported surface is also available from the package root.
"""

from ..contracts import (
    PsfMetricPrediction,
)
from ._core import zenith_angle_to_airmass
from .psf import (
    PsfComparisonSummary,
    PsfInterpolator,
    PsfPrediction,
    PsfSamples,
    build_psf_interpolator,
    compare_psf_interpolator,
    load_psf_interpolator,
    save_psf_interpolator,
)
from .psf_metrics import (
    PsfMetricComparisonSummary,
    PsfMetricInterpolator,
    PsfMetricSamples,
    build_psf_metric_interpolator,
    build_psf_metric_interpolator_from_psfs,
    compare_psf_metric_interpolator,
    load_psf_metric_interpolator,
    measure_psf_metrics,
    save_psf_metric_interpolator,
)

# Keep canonical exports in scientific workflow order.
__all__ = [  # noqa: RUF022
    # PSF images
    "PsfSamples",
    "PsfInterpolator",
    "PsfPrediction",
    "build_psf_interpolator",
    "PsfComparisonSummary",
    "compare_psf_interpolator",
    "save_psf_interpolator",
    "load_psf_interpolator",
    # PSF metrics
    "PsfMetricSamples",
    "PsfMetricInterpolator",
    "PsfMetricPrediction",
    "build_psf_metric_interpolator_from_psfs",
    "measure_psf_metrics",
    "build_psf_metric_interpolator",
    "PsfMetricComparisonSummary",
    "compare_psf_metric_interpolator",
    "save_psf_metric_interpolator",
    "load_psf_metric_interpolator",
    # Coordinate conversion
    "zenith_angle_to_airmass",
]
