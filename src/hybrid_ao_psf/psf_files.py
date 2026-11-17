"""Supply existing PSF images through a caller-owned file reader."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from ao_stats import PsfMetadata
from astropy import units as u
from numpy.typing import DTypeLike

from ._units import quantity_value
from ._validation import coordinate_precision, validate_psf_dtype
from .contracts import PsfPrediction
from .interpolation._core import field_coordinates, require_positive_scalar
from .interpolation.psf import PsfSamples, _validate_and_convert_psf_samples


def _coordinate_matches(
    stored: np.ndarray, requested: np.ndarray | float, precision: np.dtype
) -> np.ndarray:
    """Match stored coordinates with allowance for their numerical rounding."""
    tolerance = np.maximum(1e-10, 2 * np.finfo(precision).eps * np.maximum(1, np.abs(stored)))
    return np.abs(stored - requested) <= tolerance


def _check_common_sampling(reference: PsfPrediction, current: PsfPrediction) -> None:
    """Require one image shape, pixel scale, and aperture within a prediction."""
    if (
        reference.psfs.shape[1:] != current.psfs.shape[1:]
        or reference.pixel_scale != current.pixel_scale
        or reference.metadata.tel_diameter != current.metadata.tel_diameter
        or not np.array_equal(reference.metadata.tel_pupil, current.metadata.tel_pupil)
    ):
        raise ValueError("Requested PSFs must share image shape, pixel scale, and aperture.")


@dataclass(frozen=True, kw_only=True)
class PsfFileProvider:
    """Read matching PSFs from files for each prediction.

    The caller supplies the file list and a reader that returns PsfSamples.
    Each prediction reads those files, matches wavelength, zenith angle, and
    field coordinates, and selects the existing images. Matching allows only
    numerical rounding. Images retain their stored flux and orientation.

    Attributes:
        files: Nonempty sequence of PSF file paths, resolved at construction.
        reader: Callable accepting a Path and returning PsfSamples. It owns
            file-format conversion, angular units, and image orientation.
            Samples must include zenith angles for every cube.
        dtype: Image and output precision, float32 by default or float64.
            Hybrid applies its request precision internally before reading files.
    """

    files: Sequence[Path]
    reader: Callable[[Path], PsfSamples]
    dtype: DTypeLike = np.float32

    def __post_init__(self) -> None:
        if isinstance(self.files, (str, Path)):
            raise TypeError("files must be a sequence of PSF paths.")
        files = tuple(Path(path).expanduser().resolve() for path in self.files)
        if not files or len(set(files)) != len(files):
            raise ValueError("Provide a nonempty sequence of distinct PSF files.")
        if not callable(self.reader):
            raise TypeError("reader must return PsfSamples for a supplied Path.")
        object.__setattr__(self, "files", files)
        object.__setattr__(self, "dtype", validate_psf_dtype(self.dtype))

    def _with_dtype(self, dtype: DTypeLike) -> PsfFileProvider:
        """Select output precision without reading files or altering this provider."""
        return replace(self, dtype=validate_psf_dtype(dtype))

    def _read_matching_psfs(
        self, coordinates: np.ndarray, wavelength_um: float, angle: float
    ) -> Iterator[tuple[np.ndarray, PsfPrediction]]:
        """Read each file and yield images whose saved coordinates match."""
        for path in self.files:
            if not path.is_file():
                raise FileNotFoundError(f"PSF file not found: {path}")
            source = self.reader(path)
            if not isinstance(source, PsfSamples):
                raise TypeError("The PSF file reader must return PsfSamples.")
            precision = coordinate_precision(
                self.dtype, source.x, source.y, source.wavelength, source.zenith_angle
            )
            samples = _validate_and_convert_psf_samples(source, dtype=self.dtype)
            if samples.zenith_angle is None:
                raise ValueError(f"{path}: PSF cubes require zenith_angle.")
            cubes = _coordinate_matches(
                samples.wavelength.to_value(u.um), wavelength_um, precision
            ) & _coordinate_matches(samples.zenith_angle.to_value(u.deg), angle, precision)
            field_matches = _coordinate_matches(
                samples.x.to_value(u.arcsec)[None, :], coordinates[:, 0, None], precision
            ) & _coordinate_matches(
                samples.y.to_value(u.arcsec)[None, :], coordinates[:, 1, None], precision
            )
            for cube in np.flatnonzero(cubes):
                requested_indices, sample_indices = np.nonzero(field_matches)
                if not len(requested_indices):
                    continue
                selected = PsfPrediction(
                    psfs=samples.psfs[cube, sample_indices],
                    pixel_scale=samples.pixel_scale[cube],
                    metadata=PsfMetadata(
                        wavelength=wavelength_um * u.um,
                        pixel_scale=samples.pixel_scale[cube],
                        tel_diameter=samples.tel_diameter,
                        tel_pupil=samples.tel_pupil,
                    ),
                    dtype=self.dtype,
                )
                yield requested_indices, selected

    def evaluate(
        self,
        *,
        zenith_angle: u.Quantity,
        wavelength: u.Quantity,
        x: u.Quantity,
        y: u.Quantity,
    ) -> PsfPrediction:
        """Select saved PSFs and return them in the requested field order.

        Args:
            zenith_angle: Scalar angle in the interval [0, 90) degrees.
            wavelength: Positive scalar wavelength.
            x: Angular field x-coordinates.
            y: Angular field y-coordinates paired with x.

        Returns:
            Existing images at the selected precision, with a shared pixel
            scale and aperture. Repeated requests return repeated images.

        Raises:
            FileNotFoundError: If a supplied file is unavailable.
            TypeError: If coordinates have invalid types or the reader does
                not return PsfSamples.
            ValueError: If a request has no unique saved match, samples lack
                zenith angles, or selected images have incompatible sampling.
        """
        coordinates = field_coordinates(x, y)
        wavelength_um = require_positive_scalar(
            quantity_value(wavelength, u.um, label="wavelength"), label="wavelength"
        )
        angle = quantity_value(zenith_angle, u.deg, label="zenith_angle")
        if angle.ndim != 0 or not np.isfinite(angle) or not 0 <= angle < 90:
            raise ValueError("zenith_angle must be scalar in [0, 90) degrees.")

        images: list[np.ndarray | None] = [None] * len(coordinates)
        reference = None
        for requested_indices, selected in self._read_matching_psfs(
            coordinates, wavelength_um, float(angle)
        ):
            if reference is None:
                reference = selected
            else:
                _check_common_sampling(reference, selected)
            for index, image in zip(requested_indices, selected.psfs, strict=True):
                if images[index] is not None:
                    raise ValueError(
                        f"Multiple saved PSFs match field coordinate {coordinates[index]} arcsec."
                    )
                images[index] = image

        missing = [index for index, image in enumerate(images) if image is None]
        if missing:
            raise ValueError(
                f"No saved PSF at wavelength {wavelength_um} um, zenith angle "
                f"{float(angle)} deg, and field coordinates {coordinates[missing]} arcsec."
            )
        return PsfPrediction(
            psfs=np.stack(images),
            pixel_scale=reference.pixel_scale,
            metadata=reference.metadata,
            dtype=self.dtype,
        )
