"""Private MASTSEL lifetime, baseline capture and fixed-gain noise recovery."""

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, replace
from numbers import Real
from pathlib import Path
from typing import Any

import numpy as np
from astropy import units as u

from .contracts import (
    HybridCtotContext,
    HybridMastselStatistics,
    HybridRequest,
    NgsHoMetricPrediction,
)


@dataclass
class _MastselCapture:
    covariance: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
    reconstructor: np.ndarray | None = None


def _cpu_array(value: object) -> np.ndarray:
    """Copy a native NumPy/CuPy numerical return to host memory."""
    if hasattr(value, "get"):
        value = value.get()
    return np.array(value, dtype=float, copy=True)


def _load_mavis_lo() -> Any:
    try:
        from mastsel import MavisLO
    except Exception as exc:  # pragma: no cover - environment failure.
        raise RuntimeError(
            "MASTSEL is not importable; cannot run Hybrid AO PSF."
        ) from exc
    return MavisLO


def _prepare_mastsel_context(
    directory: str,
    request: HybridRequest,
    metrics: NgsHoMetricPrediction,
    effective_ini: str,
    ngs_frame_rate: u.Quantity,
    ngs_flux: u.Quantity,
) -> HybridCtotContext:
    """Construct the one runtime inside the caller-owned temporary directory."""
    ini_path = Path(directory) / "simulation.ini"
    ini_path.write_text(effective_ini, encoding="utf-8")
    mlo = _load_mavis_lo()(str(ini_path.parent), ini_path.stem, verbose=False)
    if getattr(mlo, "error", False):
        raise RuntimeError("MASTSEL failed to load the effective Hybrid configuration.")
    mas2nm = getattr(mlo, "mas2nm", None)
    if mas2nm is None:
        raise ValueError("MASTSEL did not expose mas2nm conversion.")
    return HybridCtotContext(
        mastsel=mlo,
        mastsel_statistics=None,
        angle_to_wavefront_scale=float(mas2nm) * u.nm / u.mas,
        request=request,
        ngs_metrics=metrics,
        ngs_flux=ngs_flux,
        ngs_frame_rate=ngs_frame_rate,
        effective_mastsel_ini=effective_ini,
    )


def _compute_mastsel_ctot(context: HybridCtotContext) -> u.Quantity:
    """Evaluate baseline Ctot once with the prepared runtime and resolved inputs."""
    request, metrics = context.request, context.ngs_metrics
    science = np.column_stack(
        (request.science_x.to_value(u.arcsec), request.science_y.to_value(u.arcsec))
    )
    ngs = np.column_stack(
        (request.ngs_x.to_value(u.arcsec), request.ngs_y.to_value(u.arcsec))
    )
    return (
        np.asarray(
            context.mastsel.computeTotalResidualMatrix(
                science,
                ngs,
                context.ngs_flux.to_value(u.photon / u.s),
                context.ngs_frame_rate.to_value(u.Hz),
                metrics.sr.to_value(u.one),
                metrics.ee.to_value(u.one),
                metrics.fwhm.to_value(u.mas),
                aNGS_FWHM_DL_mas=None,
                doAll=True,
            ),
            dtype=float,
        )
        * u.nm**2
    )


@contextmanager
def _capture_mastsel(context: HybridCtotContext) -> Iterator[_MastselCapture]:
    """Wrap only this runtime's two baseline methods and restore their bindings."""
    model = context.mastsel
    names = ("computeCovMatrices", "buildReconstuctor2")
    original = {name: getattr(model, name) for name in names}
    bound = {name: model.__dict__[name] for name in names if name in model.__dict__}
    captured = _MastselCapture()

    def covariance(*args: object, **kwargs: object) -> object:
        result = original["computeCovMatrices"](*args, **kwargs)
        if captured.covariance is not None:
            raise RuntimeError("Expected one baseline MASTSEL covariance calculation.")
        captured.covariance = tuple(_cpu_array(value) for value in result)
        return result

    def reconstruction(*args: object, **kwargs: object) -> object:
        result = original["buildReconstuctor2"](*args, **kwargs)
        if captured.reconstructor is not None:
            raise RuntimeError(
                "Expected one baseline MASTSEL reconstructor calculation."
            )
        captured.reconstructor = _cpu_array(result[0])
        return result

    try:
        model.computeCovMatrices = covariance
        model.buildReconstuctor2 = reconstruction
        yield captured
    finally:
        for name in names:
            if name in bound:
                setattr(model, name, bound[name])
            elif name in model.__dict__:
                delattr(model, name)


def _measurement_noise(context: HybridCtotContext) -> u.Quantity | None:
    """Recover filtered noise only; retained nr also includes atmospheric servo.

    Reuse baseline bias and centroid variance. The grid, transfer function and
    normalization match MASTSEL's fixed-gain residual calculation, without
    re-evaluating detector moments or atmospheric spectra. Both axes use avar[0],
    as in baseline MASTSEL. Non-fixed gain has no supported noise-only output.
    """
    model = context.mastsel
    gain = model.LoopGain_LO
    if not isinstance(gain, Real):
        return None
    count = context.request.ngs_x.size
    if model.noNoise:
        return np.zeros((2 * count, 2 * count)) * u.nm**2
    values = []
    for index, rate in enumerate(context.ngs_frame_rate.to_value(u.Hz)):
        detector_index = index if len(model.NumberLenslets) == count else 0
        variance = float(_cpu_array(model.avar[index][0]))
        variance *= (
            model.PixelScale_LO[detector_index] ** 2 / model.N_sa_tot_LO[detector_index]
        )
        bias = float(_cpu_array(model.bias[index]))
        if not np.isfinite(bias) or bias == 0:
            raise ValueError(
                "MASTSEL sensor bias must be finite and nonzero for noise recovery."
            )
        frequency = np.linspace(0.25, rate / 2, int(2 * rate))
        if frequency.size < 2:
            raise ValueError(
                "MASTSEL noise recovery requires at least two frequency samples."
            )
        effective_gain = bias * gain
        z = np.exp(2j * np.pi * frequency / rate)
        delayed_gain = effective_gain * z ** (-model.loopDelaySteps_LO)
        hn = delayed_gain / (1 - z**-1 + delayed_gain)
        filtered = variance / bias**2 / (frequency.size - 1) * np.sum(abs(hn) ** 2)
        values.extend((filtered, filtered))
    return np.diag(values) * context.angle_to_wavefront_scale**2 * u.mas**2


def _complete_mastsel_context(
    context: HybridCtotContext, captured: _MastselCapture
) -> HybridCtotContext:
    """Bind snapshots and centre covariance, without repeating the field baseline."""
    if captured.covariance is None or captured.reconstructor is None:
        raise RuntimeError(
            "MASTSEL baseline did not expose covariance and reconstructor returns."
        )
    request = context.request
    count, size = request.ngs_x.size, request.science_x.size
    _, cas, css = captured.covariance
    cas = cas.reshape(size, 2, 2 * count)
    center_indices = np.flatnonzero(
        (request.science_x.to_value(u.arcsec) == 0)
        & (request.science_y.to_value(u.arcsec) == 0)
    )
    centre = None
    if center_indices.size:
        centre = cas[center_indices[0]]
    elif count > 1:
        ngs = np.column_stack(
            (request.ngs_x.to_value(u.arcsec), request.ngs_y.to_value(u.arcsec))
        )
        _, centre, _ = context.mastsel.computeCovMatrices(np.zeros((1, 2)), ngs)
        centre = _cpu_array(centre)
    statistics = HybridMastselStatistics(
        ngs_covariance=css * u.nm**2,
        science_ngs_covariance=cas * u.nm**2,
        centre_ngs_covariance=None if centre is None else centre * u.nm**2,
        reconstructor=captured.reconstructor.reshape(size, 2, 2 * count),
        measurement_noise=_measurement_noise(context),
    )
    return replace(context, mastsel_statistics=statistics)
