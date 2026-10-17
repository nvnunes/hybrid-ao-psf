"""Scientific precision through blur, metrics and standalone persistence."""

from dataclasses import replace

import h5py
import numpy as np
import pytest
from ao_stats import compute_psf_fwhm

from hybrid_ao_psf import (
    ScienceHoPsfPrediction,
    _mastsel,
    load_result,
    save_result,
    simulate,
)

from .test_contracts import make_result
from .test_engine import _providers, _request, _ScienceProvider


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_complete_blur_statistics_and_result_keep_selected_values(
    monkeypatch, tmp_path, dtype
):
    class Mavis:
        error = False
        mas2nm = 2.0

        def __init__(self, *args, **kwargs):
            pass

        def computeTotalResidualMatrix(self, *args, **kwargs):
            return np.broadcast_to([[32.0, 4.0], [4.0, 20.0]], (2, 2, 2)).copy()

    monkeypatch.setattr(_mastsel, "_load_mavis_lo", lambda: Mavis)
    old, ngs = _providers()
    y, x = np.mgrid[-4:5, -4:5]
    source = np.broadcast_to(np.exp(-(x * x + y * y) / 5.0), (2, 9, 9)).copy()
    source[:, 4, 4] += 1e-8
    request = replace(_request(), dtype=dtype)
    science = _ScienceProvider(
        ScienceHoPsfPrediction(
            source, old.prediction.pixel_scale, old.prediction.metadata, dtype=dtype
        )
    )
    result = simulate(request, science, ngs)
    # Independent NumPy double FFT reference, with explicitly rounded inputs
    # in float32 mode; no engine OTF or padding helpers are used here.
    cov = (
        np.array([[32.0, 4.0], [4.0, 20.0]], dtype=dtype)
        / dtype(2.0) ** 2
        / dtype(8.0) ** 2
    )
    fx, fy = np.meshgrid(np.fft.fftfreq(18), np.fft.fftfreq(18))
    otf = np.exp(
        -2
        * np.pi**2
        * (cov[0, 0] * fx**2 + 2 * cov[0, 1] * fx * fy + cov[1, 1] * fy**2)
    )
    expected = []
    for image in source.astype(dtype):
        padded = np.pad(image.astype(np.float64), ((4, 5), (4, 5)))
        expected.append(
            np.maximum(np.fft.ifft2(np.fft.fft2(padded) * otf).real[4:13, 4:13], 0)
        )
    np.testing.assert_allclose(
        result.psfs,
        expected,
        rtol=8e-7 if dtype == np.float32 else 2e-14,
        atol=2e-7 if dtype == np.float32 else 2e-15,
    )
    assert result.dtype == dtype and result.psfs.dtype == dtype
    for quantity in (
        result.ctot_wavefront,
        result.ctot_angle,
        result.jitter,
        result.ngs_flux,
        result.metadata.pixel_scale,
        result.ngs_metrics.fwhm,
    ):
        assert quantity.dtype == dtype
    assert result.diagnostics.pre_blur_psfs.dtype == dtype
    if dtype == np.float64:
        assert result.diagnostics.pre_blur_psfs[0, 4, 4] == 1.0 + 1e-8
        assert result.diagnostics.pre_blur_psfs[0, 4, 4] != np.float32(1.0 + 1e-8)
    metric = compute_psf_fwhm(result.psfs, result.metadata, dtype=dtype)
    assert metric.dtype == dtype
    path = save_result(tmp_path / "result.h5", result)
    loaded = load_result(path)
    assert loaded.dtype == dtype
    np.testing.assert_array_equal(loaded.psfs, result.psfs)
    np.testing.assert_array_equal(
        compute_psf_fwhm(loaded.psfs, loaded.metadata, dtype=dtype), metric
    )


def test_legacy_mixed_result_keeps_every_stored_floating_array(tmp_path):
    result = replace(make_result(), dtype=np.float64)
    path = save_result(tmp_path / "legacy.h5", result)
    stored = {}
    with h5py.File(path, "r+") as root:
        root.attrs["schema_version"] = 1
        del root.attrs["precision"]
        ee = root["result/ngs_metrics/ee"][...].astype(np.float32)
        del root["result/ngs_metrics/ee"]
        ds = root.create_dataset("result/ngs_metrics/ee", data=ee)
        ds.attrs["unit"] = ""
        pixels = root["result/psfs"][...].astype(np.float32)
        del root["result/psfs"]
        root.create_dataset("result/psfs", data=pixels)

        def snapshot(name, item):
            if isinstance(item, h5py.Dataset) and item.dtype.kind == "f":
                stored[name] = item[...]

        root.visititems(snapshot)
    loaded = load_result(path)
    assert loaded.dtype is None
    assert loaded.psfs.dtype == np.float32 and loaded.ctot_angle.dtype == np.float64
    rewritten = save_result(tmp_path / "rewritten.h5", loaded)
    with h5py.File(rewritten) as root:
        for name, values in stored.items():
            assert root[name].dtype == values.dtype
            np.testing.assert_array_equal(root[name][...], values)


def test_result_declared_precision_mismatch_is_rejected(tmp_path):
    path = save_result(tmp_path / "bad.h5", make_result())
    with h5py.File(path, "r+") as root:
        root.attrs["precision"] = "float64"
    with pytest.raises(ValueError, match="precision mismatch"):
        load_result(path)
