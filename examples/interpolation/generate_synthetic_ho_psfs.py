"""Generate synthetic science and NGS HO PSFs from Airy–Moffat models.

The core fraction and halo width depend on airmass. The core fraction
also depends on wavelength. Each saved image is centred and normalized
to unit sum.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from astropy import units as u
from psf_fits import PsfCube, write_psf_cube
from scipy.special import j1

TELESCOPE_DIAMETER_M = 8.0
CENTRAL_OBSTRUCTION_DIAMETER_M = 1.3
PUPIL_SAMPLES_ACROSS_DIAMETER = 128
FIELD_RADIUS_ARCSEC = 60.0
GRID_POINTS_ACROSS_DIAMETER = 7
ZENITH_ANGLES_DEG = (0, 20, 30, 40)
SCIENCE_WAVELENGTHS_UM = (1.215, 1.654, 2.179)
NGS_WAVELENGTH_UM = 0.710
SCIENCE_PSF_PIXELS = 121
NGS_PSF_PIXELS = 401


@dataclass(frozen=True)
class PsfModel:
    """Airy–Moffat parameters with radial values at field centre and edge."""

    reference_wavelength_um: float
    zenith_core_fractions: tuple[float, float]
    core_airmass_powers: tuple[float, float]
    zenith_halo_fwhms_mas: tuple[float, float]
    halo_airmass_power: float
    moffat_beta: float

    def evaluate(
        self,
        field_radius_arcsec: np.ndarray,
        zenith_angle_deg: float,
        wavelength_um: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return the core fraction and halo FWHM at each field radius."""
        airmass = 1 / np.cos(np.deg2rad(zenith_angle_deg))
        phase_scale = (
            airmass ** np.asarray(self.core_airmass_powers)
            * (self.reference_wavelength_um / wavelength_um) ** 2
        )
        core_fractions = np.asarray(self.zenith_core_fractions) ** phase_scale
        halo_fwhms_mas = (
            np.asarray(self.zenith_halo_fwhms_mas) * airmass**self.halo_airmass_power
        )
        field_limits = (0.0, FIELD_RADIUS_ARCSEC)
        core_fraction = np.interp(field_radius_arcsec, field_limits, core_fractions)
        halo_fwhm_mas = np.interp(field_radius_arcsec, field_limits, halo_fwhms_mas)
        return core_fraction, halo_fwhm_mas


SCIENCE_MODEL = PsfModel(
    reference_wavelength_um=1.654,
    zenith_core_fractions=(0.65, 0.30),
    core_airmass_powers=(1.2, 1.5),
    zenith_halo_fwhms_mas=(90.0, 90.0),
    halo_airmass_power=0.4,
    moffat_beta=1.25,
)

NGS_MODEL = PsfModel(
    reference_wavelength_um=NGS_WAVELENGTH_UM,
    zenith_core_fractions=(0.0075, 0.0),
    core_airmass_powers=(1.1, 0.0),
    zenith_halo_fwhms_mas=(150.0, 190.0),
    halo_airmass_power=1.0,
    moffat_beta=1.85,
)


def main(output_dir: Path) -> None:
    """Write the science and NGS FITS cubes on their observing grids."""
    science_dir = output_dir / "science-psfs"
    ngs_dir = output_dir / "ngs-psfs"
    science_dir.mkdir(parents=True, exist_ok=True)
    ngs_dir.mkdir(parents=True, exist_ok=True)

    pupil = _annular_pupil()
    positions_arcsec = _field_grid()
    for za in ZENITH_ANGLES_DEG:
        for wavelength in SCIENCE_WAVELENGTHS_UM:
            science_cube = _generate_psf_cube(
                pupil,
                positions_arcsec=positions_arcsec,
                zenith_angle_deg=za,
                wavelength_um=wavelength,
                psf_side_pixels=SCIENCE_PSF_PIXELS,
                model=SCIENCE_MODEL,
            )
            filename = f"science-za{za:g}-wl{wavelength:.3f}.fits"
            write_psf_cube(science_dir / filename, science_cube, overwrite=True)

        ngs_cube = _generate_psf_cube(
            pupil,
            positions_arcsec=positions_arcsec,
            zenith_angle_deg=za,
            wavelength_um=NGS_WAVELENGTH_UM,
            psf_side_pixels=NGS_PSF_PIXELS,
            model=NGS_MODEL,
        )
        write_psf_cube(ngs_dir / f"ngs-za{za:g}.fits", ngs_cube, overwrite=True)


def _field_grid() -> np.ndarray:
    """Sample the rectangular field grid shared by both libraries."""
    axis_arcsec = np.linspace(
        -FIELD_RADIUS_ARCSEC, FIELD_RADIUS_ARCSEC, GRID_POINTS_ACROSS_DIAMETER
    )
    x_arcsec, y_arcsec = np.meshgrid(axis_arcsec, axis_arcsec)
    return np.column_stack((x_arcsec.ravel(), y_arcsec.ravel()))


def _annular_pupil() -> np.ndarray:
    """Make the shared pupil mask for the two synthetic libraries."""
    outer_sample = 1.0 - 1.0 / PUPIL_SAMPLES_ACROSS_DIAMETER
    pupil_axis = np.linspace(-outer_sample, outer_sample, PUPIL_SAMPLES_ACROSS_DIAMETER)
    pupil_x, pupil_y = np.meshgrid(pupil_axis, pupil_axis)
    radius_squared = pupil_x**2 + pupil_y**2
    obstruction_ratio = CENTRAL_OBSTRUCTION_DIAMETER_M / TELESCOPE_DIAMETER_M
    transmitting = (radius_squared <= 1.0) & (radius_squared >= obstruction_ratio**2)
    return transmitting.astype(np.float32)


def _generate_psf_cube(
    pupil: np.ndarray,
    *,
    positions_arcsec: np.ndarray,
    zenith_angle_deg: float,
    wavelength_um: float,
    psf_side_pixels: int,
    model: PsfModel,
) -> PsfCube:
    x_arcsec, y_arcsec = positions_arcsec.T
    field_radius_arcsec = np.hypot(x_arcsec, y_arcsec)
    pixel_scale_mas = _pixel_scale_mas(wavelength_um)
    core_fraction, halo_fwhm_mas = model.evaluate(
        field_radius_arcsec, zenith_angle_deg, wavelength_um
    )
    psfs = _render_psfs(
        wavelength_um=wavelength_um,
        pixel_scale_mas=pixel_scale_mas,
        psf_side_pixels=psf_side_pixels,
        core_fraction=core_fraction,
        halo_fwhm_mas=halo_fwhm_mas,
        moffat_beta=model.moffat_beta,
    )
    return PsfCube(
        psfs=psfs,
        x_arcsec=x_arcsec,
        y_arcsec=y_arcsec,
        pupil=pupil,
        wavelength_um=wavelength_um,
        pixel_scale_mas=pixel_scale_mas,
        zenith_angle_deg=zenith_angle_deg,
        tel_diameter_m=TELESCOPE_DIAMETER_M,
    )


def _pixel_scale_mas(wavelength_um: float) -> float:
    """Sample the diffraction scale λ/D with four pixels."""
    wavelength_m = (wavelength_um * u.um).to_value(u.m)
    pixel_scale_rad = wavelength_m / (4 * TELESCOPE_DIAMETER_M)
    return (pixel_scale_rad * u.rad).to_value(u.mas)


def _render_psfs(
    *,
    wavelength_um: float,
    pixel_scale_mas: float,
    psf_side_pixels: int,
    core_fraction: np.ndarray,
    halo_fwhm_mas: np.ndarray,
    moffat_beta: float,
) -> np.ndarray:
    """Mix full-flux components and normalize each finished stamp."""
    pixel_offsets = np.arange(psf_side_pixels) - (psf_side_pixels - 1) / 2
    x_pixels, y_pixels = np.meshgrid(pixel_offsets, pixel_offsets)
    radius_mas = np.hypot(x_pixels, y_pixels) * pixel_scale_mas

    core = _airy_core(radius_mas, wavelength_um)
    halo = _moffat_halo(radius_mas, halo_fwhm_mas[:, None, None], moffat_beta)
    core_fraction = core_fraction[:, None, None]
    psfs = core_fraction * core + (1.0 - core_fraction) * halo
    return (psfs / psfs.sum(axis=(-2, -1), keepdims=True)).astype(np.float32)


def _airy_core(radius_mas: np.ndarray, wavelength_um: float) -> np.ndarray:
    """Evaluate the unit-flux annular-aperture diffraction pattern."""
    wavelength_m = (wavelength_um * u.um).to_value(u.m)
    scale_per_mas = np.pi * TELESCOPE_DIAMETER_M * u.mas.to(u.rad) / wavelength_m
    argument = scale_per_mas * radius_mas
    obstruction = CENTRAL_OBSTRUCTION_DIAMETER_M / TELESCOPE_DIAMETER_M
    pupil_area_fraction = 1.0 - obstruction**2
    amplitude_numerator = 2 * (j1(argument) - obstruction * j1(obstruction * argument))
    amplitude = np.divide(
        amplitude_numerator,
        argument * pupil_area_fraction,
        out=np.ones_like(argument),
        where=argument != 0,
    )
    return amplitude**2 * pupil_area_fraction * scale_per_mas**2 / (4 * np.pi)


def _moffat_halo(
    radius_mas: np.ndarray, fwhm_mas: np.ndarray, beta: float
) -> np.ndarray:
    """Evaluate the unit-flux Moffat profile at the specified widths."""
    radius_scale_mas = fwhm_mas / (2 * np.sqrt(2 ** (1 / beta) - 1))
    peak = (beta - 1) / (np.pi * radius_scale_mas**2)
    return peak * (1 + (radius_mas / radius_scale_mas) ** 2) ** (-beta)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Output directory for the science-psfs and ngs-psfs libraries.",
    )
    main(parser.parse_args().output_dir)
