import stat
from dataclasses import replace
from pathlib import Path

import h5py
import numpy as np
import pytest
from ao_core.io import collect_schema_issues, read_result
from astropy import units as u

from hybrid_ao_psf import (
    load_result,
    result_io,
    save_result,
    simulate,
)

from .test_contracts import make_result


def save_simulation(path, request, science, *, lo_model):
    """Write the result from a publicly configured in-memory simulation."""
    result = simulate(request, science, lo_model=lo_model)
    save_result(path, result)
    return result


def test_result_hdf5_round_trip(tmp_path: Path) -> None:
    expected = make_result()
    path = save_result(tmp_path / "nested" / "result.h5", expected)

    with h5py.File(path, "r") as root:
        assert not collect_schema_issues(root)
        assert read_result(root)["stats"]["jitter"].unit == u.mas
        assert set(root) == {
            "setup",
            "options",
            "simulation",
            "meta",
            "psfs",
            "stats",
            "diagnostics",
        }
        np.testing.assert_array_equal(root["setup/sci_r"], [0.0, 1.0])
        np.testing.assert_array_equal(root["setup/sci_theta"], [0.0, 0.0])
        assert root["options/wavelength"].attrs["units"] == "um"
        assert root["simulation/precision"].asstr()[()] == "float32"
        assert root["meta/tel_pupil"].attrs["units"] == "1"
        assert root["psfs/data"].shape == (2, 4, 4)
        assert root["stats/jitter"].attrs["units"] == "mas"
        rates = root["diagnostics/mastsel/frame_rate"]
        assert rates.dtype == expected.psfs.dtype and rates.attrs["units"] == "Hz"
        assert rates.shape == () and rates[()] == 500.0
        assert "diagnostics/covariance/psd_valid_mask" in root
        assert "diagnostics/mastsel/effective_ini" in root
        assert "diagnostics/psfs_ho" in root

    actual = load_result(path)
    assert actual.diagnostics.mastsel.frame_rate.isscalar
    assert (
        actual.diagnostics.mastsel.frame_rate == expected.diagnostics.mastsel.frame_rate
    )
    np.testing.assert_array_equal(actual.psfs, expected.psfs)
    np.testing.assert_array_equal(
        actual.metadata.tel_pupil, expected.metadata.tel_pupil
    )
    np.testing.assert_allclose(
        actual.diagnostics.mastsel.ngs_metrics.fwhm.to_value(u.mas),
        expected.diagnostics.mastsel.ngs_metrics.fwhm.value,
    )
    np.testing.assert_array_equal(
        actual.diagnostics.covariance.psd_clipped,
        expected.diagnostics.covariance.psd_clipped,
    )
    assert (
        actual.diagnostics.mastsel.effective_ini
        == expected.diagnostics.mastsel.effective_ini
    )
    assert actual.diagnostics is not None
    assert expected.diagnostics is not None
    np.testing.assert_array_equal(
        actual.diagnostics.covariance.psd_valid_mask,
        expected.diagnostics.covariance.psd_valid_mask,
    )
    np.testing.assert_array_equal(
        actual.diagnostics.covariance.ctot, expected.diagnostics.covariance.ctot
    )
    np.testing.assert_array_equal(
        actual.diagnostics.psfs_ho,
        expected.diagnostics.psfs_ho,
    )
    assert not actual.psfs.flags.writeable


def test_load_result_rejects_v6_layout_without_polar_grid(tmp_path: Path) -> None:
    path = save_result(tmp_path / "result.h5", make_result(diagnostics=False))
    with h5py.File(path, "r+") as root:
        root.attrs["schema"] = "hybrid-ao-psf-result"
        root.attrs["schema_version"] = 6
        del root["setup/sci_r"]
        del root["setup/sci_theta"]
        root["setup"].create_dataset("science_positions", data=[[0, 0], [1, 0]])

    with pytest.raises(ValueError, match="science"):
        load_result(path)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_polar_storage_preserves_order_and_cartesian_positions(tmp_path, dtype):
    positions = np.array([[0, 0], [-14, -0.24]], dtype=dtype) * u.arcsec
    result = replace(
        make_result(diagnostics=False),
        positions=positions,
        psfs=np.arange(1, 33, dtype=dtype).reshape(2, 4, 4),
    )
    path = save_result(tmp_path / "result.h5", result)
    loaded = load_result(path)
    assert loaded.positions.dtype == dtype
    tolerance = 4 * np.finfo(dtype).eps * np.max(np.abs(positions.value))
    np.testing.assert_allclose(
        loaded.positions.value, positions.value, rtol=0, atol=tolerance
    )
    np.testing.assert_array_equal(loaded.psfs, result.psfs)
    np.testing.assert_array_equal(loaded.jitter, result.jitter)


def test_minimal_result_contains_no_diagnostics(tmp_path):
    expected = make_result(diagnostics=False)
    path = save_result(tmp_path / "result.h5", expected)

    with h5py.File(path, "r") as root:
        assert set(root) == {"setup", "options", "simulation", "meta", "psfs", "stats"}
        assert set(root["options"]) == {"wavelength"}
        assert root["simulation/diagnostics_level"].asstr()[()] == "none"
        assert root["simulation/extra_stat_fields/jitter"].asstr()[()] == "mas"
    actual = load_result(path)
    assert actual.diagnostics is None
    np.testing.assert_array_equal(actual.psfs, expected.psfs)
    np.testing.assert_array_equal(actual.jitter, expected.jitter)


def test_validation_result_round_trip_omits_debug_psfs(tmp_path: Path) -> None:
    expected = make_result(debug=False)

    path = save_result(tmp_path / "validation.h5", expected)

    with h5py.File(path, "r") as root:
        assert "diagnostics/covariance/psd_valid_mask" in root
        assert "diagnostics/psfs_ho" not in root
    actual = load_result(path)
    assert actual.diagnostics is not None
    assert actual.diagnostics.psfs_ho is None
    assert actual.diagnostics.mastsel.effective_ini is None


def test_load_result_rejects_jitter_inconsistent_with_ctot(tmp_path: Path) -> None:
    path = save_result(tmp_path / "result.h5", make_result())
    with h5py.File(path, "r+") as root:
        root["stats/jitter"][...] *= 2

    with pytest.raises(ValueError, match="jitter must equal"):
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

    with pytest.raises(ValueError, match="options/wavelength"):
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
