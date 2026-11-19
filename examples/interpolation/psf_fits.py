"""Read and write the FITS cubes used by the interpolation examples."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from astropy import units as u
from astropy.io import fits

from hybrid_ao_psf import PsfSamples


@dataclass(frozen=True)
class PsfCube:
    """A PSF cube at one zenith angle and wavelength, in (point, row, column) order."""

    psfs: np.ndarray
    x_arcsec: np.ndarray
    y_arcsec: np.ndarray
    pupil: np.ndarray
    wavelength_um: float
    pixel_scale_mas: float
    zenith_angle_deg: float
    tel_diameter_m: float

    def __post_init__(self) -> None:
        if self.psfs.ndim != 3:
            raise ValueError("the primary HDU must contain a 3D PSF cube")
        expected_shape = (self.psfs.shape[0],)
        if (
            self.x_arcsec.shape != expected_shape
            or self.y_arcsec.shape != expected_shape
        ):
            raise ValueError("XFIELD and YFIELD must give one coordinate per PSF")
        if self.pupil.ndim != 2:
            raise ValueError("PUPIL must contain a 2D pupil mask")


def read_psf_cube(path: Path) -> PsfCube:
    """Read one cube, keeping its PSFs and field coordinates in file order."""
    with fits.open(path, memmap=False) as hdus:
        header = hdus[0].header
        try:
            return PsfCube(
                psfs=np.asarray(hdus[0].data),
                x_arcsec=np.asarray(hdus["XFIELD"].data),
                y_arcsec=np.asarray(hdus["YFIELD"].data),
                pupil=np.asarray(hdus["PUPIL"].data),
                wavelength_um=float(header["WAVELEN"]),
                pixel_scale_mas=float(header["PIXSCALE"]),
                zenith_angle_deg=float(header["ZENITH"]),
                tel_diameter_m=float(header["TELDIAM"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"{path}: invalid PSF FITS file ({exc}).") from exc


def read_psf_samples(path: Path) -> PsfSamples:
    """Read one FITS cube with physical quantities for Hybrid PSF operations."""
    cube = read_psf_cube(path)
    field_order = np.lexsort((cube.x_arcsec, cube.y_arcsec))
    return PsfSamples(
        psfs=cube.psfs[field_order],
        x=cube.x_arcsec[field_order] * u.arcsec,
        y=cube.y_arcsec[field_order] * u.arcsec,
        wavelength=cube.wavelength_um * u.um,
        pixel_scale=cube.pixel_scale_mas * u.mas,
        zenith_angle=cube.zenith_angle_deg * u.deg,
        tel_diameter=cube.tel_diameter_m * u.m,
        tel_pupil=cube.pupil * u.one,
    )


def write_psf_cube(path: Path, cube: PsfCube, *, overwrite: bool = False) -> None:
    """Write one cube with its coordinate vectors and pupil mask."""
    header = fits.Header(
        [
            ("WAVELEN", cube.wavelength_um, "[um] Wavelength"),
            ("PIXSCALE", cube.pixel_scale_mas, "[mas/pixel] Image pixel scale"),
            ("ZENITH", cube.zenith_angle_deg, "[deg] Zenith angle"),
            ("TELDIAM", cube.tel_diameter_m, "[m] Telescope diameter"),
        ]
    )
    field_header = fits.Header({"BUNIT": "arcsec"})
    fits.HDUList(
        [
            fits.PrimaryHDU(cube.psfs, header=header),
            fits.ImageHDU(cube.x_arcsec, header=field_header, name="XFIELD"),
            fits.ImageHDU(cube.y_arcsec, header=field_header, name="YFIELD"),
            fits.ImageHDU(cube.pupil, name="PUPIL"),
        ]
    ).writeto(path, overwrite=overwrite)
