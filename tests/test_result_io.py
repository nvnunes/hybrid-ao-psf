import stat
from dataclasses import replace
from pathlib import Path

import h5py
import numpy as np
import pytest
from astropy import units as u

from hybrid_ao_psf import (
    RESULT_SCHEMA,
    RESULT_SCHEMA_VERSION,
    load_result,
    result_io,
    save_result,
)

from .test_contracts import make_result


def test_result_hdf5_round_trip(tmp_path: Path) -> None:
    expected = make_result()
    path = save_result(tmp_path / "nested" / "result.h5", expected)

    with h5py.File(path, "r") as root:
        assert root.attrs["schema"] == RESULT_SCHEMA
        assert root.attrs["schema_version"] == RESULT_SCHEMA_VERSION
        assert "configuration/effective_mastsel_ini" in root
        assert "diagnostics/psd_valid_mask" in root
        assert "diagnostics/ctot_angle_trace_mean" in root
        assert "diagnostics/ctot_angle_determinant_mean" in root
        assert "diagnostics/pre_blur_psfs" in root

    actual = load_result(path)
    np.testing.assert_array_equal(actual.psfs, expected.psfs)
    np.testing.assert_array_equal(
        actual.metadata.tel_pupil, expected.metadata.tel_pupil
    )
    np.testing.assert_allclose(
        actual.ctot_wavefront.to_value(u.nm**2), expected.ctot_wavefront.value
    )
    np.testing.assert_allclose(
        actual.ngs_metrics.fwhm.to_value(u.mas), expected.ngs_metrics.fwhm.value
    )
    np.testing.assert_array_equal(actual.psd_clipped, expected.psd_clipped)
    assert actual.effective_mastsel_ini == expected.effective_mastsel_ini
    assert dict(actual.provenance) == dict(expected.provenance)
    assert actual.diagnostics is not None
    assert expected.diagnostics is not None
    np.testing.assert_array_equal(
        actual.diagnostics.psd_valid_mask,
        expected.diagnostics.psd_valid_mask,
    )
    assert (
        actual.diagnostics.ctot_angle_trace_mean
        == expected.diagnostics.ctot_angle_trace_mean
    )
    assert (
        actual.diagnostics.ctot_angle_determinant_mean
        == expected.diagnostics.ctot_angle_determinant_mean
    )
    np.testing.assert_array_equal(
        actual.diagnostics.pre_blur_psfs,
        expected.diagnostics.pre_blur_psfs,
    )
    assert not actual.psfs.flags.writeable


def test_correction_identity_round_trips_without_runtime_state(tmp_path):
    expected = replace(
        make_result(),
        provenance={"ctot_corrections": "example", "fit_path": "/fits/fit.json"},
    )
    path = save_result(tmp_path / "corrected.h5", expected)
    actual = load_result(path)
    assert dict(actual.provenance) == dict(expected.provenance)
    with h5py.File(path) as root:
        assert "mastsel_statistics" not in root and "ctot_correction" not in root


def test_magnitude_provenance_round_trip(tmp_path: Path) -> None:
    expected = replace(
        make_result(),
        ngs_magnitude=np.array([12.0, 15.0]) * u.mag,
        ngs_magnitude_zeropoint=6_277_173_913.043478 * u.photon / (u.m**2 * u.s),
    )
    path = save_result(tmp_path / "magnitude.h5", expected)
    with h5py.File(path) as root:
        assert "result/ngs/magnitude" in root
        assert "result/ngs/magnitude_zeropoint" in root
    actual = load_result(path)
    np.testing.assert_array_equal(
        actual.ngs_magnitude.value, expected.ngs_magnitude.value
    )
    assert actual.ngs_magnitude_zeropoint == expected.ngs_magnitude_zeropoint


def test_direct_flux_result_omits_magnitude_provenance(tmp_path: Path) -> None:
    path = save_result(tmp_path / "flux.h5", make_result())
    with h5py.File(path) as root:
        assert "result/ngs/magnitude" not in root
        assert "result/ngs/magnitude_zeropoint" not in root
    actual = load_result(path)
    assert actual.ngs_magnitude is None
    assert actual.ngs_magnitude_zeropoint is None


def test_validation_result_round_trip_omits_debug_psfs(tmp_path: Path) -> None:
    expected = make_result()
    assert expected.diagnostics is not None
    expected = replace(
        expected,
        diagnostics=replace(expected.diagnostics, pre_blur_psfs=None),
    )

    path = save_result(tmp_path / "validation.h5", expected)

    with h5py.File(path, "r") as root:
        assert "diagnostics/psd_valid_mask" in root
        assert "diagnostics/pre_blur_psfs" not in root
    actual = load_result(path)
    assert actual.diagnostics is not None
    assert actual.diagnostics.pre_blur_psfs is None


def test_load_result_rejects_diagnostics_inconsistent_with_ctot(
    tmp_path: Path,
) -> None:
    path = save_result(tmp_path / "result.h5", make_result())
    with h5py.File(path, "r+") as root:
        for name in (
            "ctot_angle_trace_min",
            "ctot_angle_trace_max",
            "ctot_angle_trace_mean",
        ):
            root[f"diagnostics/{name}"][...] = 2.0

    with pytest.raises(ValueError, match="ctot_angle_trace_min must match"):
        load_result(path)


def test_save_result_refuses_implicit_overwrite(tmp_path: Path) -> None:
    path = save_result(tmp_path / "result.h5", make_result())

    with pytest.raises(FileExistsError):
        save_result(path, make_result())
    save_result(path, make_result(), overwrite=True)


def test_save_result_uses_normal_permissions_and_preserves_overwrite_mode(
    tmp_path: Path,
) -> None:
    reference = tmp_path / "reference"
    reference.write_bytes(b"reference")
    expected_new_mode = stat.S_IMODE(reference.stat().st_mode)

    path = save_result(tmp_path / "result.h5", make_result())
    assert stat.S_IMODE(path.stat().st_mode) == expected_new_mode

    path.chmod(0o640)
    save_result(path, make_result(), overwrite=True)
    assert stat.S_IMODE(path.stat().st_mode) == 0o640


def test_load_result_rejects_unknown_schema(tmp_path: Path) -> None:
    path = tmp_path / "result.h5"
    with h5py.File(path, "w") as root:
        root.attrs["schema"] = "other"
        root.attrs["schema_version"] = 99

    with pytest.raises(ValueError, match="Unsupported"):
        load_result(path)


def test_failed_result_overwrite_preserves_existing_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "result.h5"
    path.write_bytes(b"existing result")

    def fail_write(*_: object) -> None:
        raise RuntimeError("write failed")

    monkeypatch.setattr(result_io, "_write_result", fail_write)
    with pytest.raises(RuntimeError, match="write failed"):
        save_result(path, make_result(), overwrite=True)

    assert path.read_bytes() == b"existing result"
    assert list(tmp_path.glob(".result.h5.*.tmp")) == []
