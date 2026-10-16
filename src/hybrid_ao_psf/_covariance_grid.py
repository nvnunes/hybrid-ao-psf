"""Rectangular construction and feedback normalization of measured tables."""

import numpy as np

from ._covariance_statistics import FUNCTIONS


def rectangular_mapping(
    positions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Map the full rectangular grid to measured nodes with nearest-node filling.

    Use one common map for both variables and all covariance blocks. Ties follow
    rectangular y/x order, matching the established science-PSF fill convention.
    """
    positions = np.asarray(positions, float)
    xs, ys = np.unique(positions[:, 0]), np.unique(positions[:, 1])
    if len(xs) < 2 or len(ys) < 2:
        raise ValueError("Covariance spatial grid needs at least two nodes per axis")
    grid = np.full((len(xs), len(ys)), -1, int)
    for index, point in enumerate(positions):
        ix, iy = np.searchsorted(xs, point[0]), np.searchsorted(ys, point[1])
        if grid[ix, iy] >= 0:
            raise ValueError("Duplicate measured field coordinate")
        grid[ix, iy] = index
    measured = grid >= 0
    valid_yx = np.argwhere(measured.T)
    for ix, iy in np.argwhere(~measured):
        distance = np.sum((valid_yx - np.array([iy, ix])) ** 2, axis=1)
        vy, vx = valid_yx[np.argmin(distance)]
        grid[ix, iy] = grid[vx, vy]
    return xs, ys, grid, measured


def rectangular_tables(native: dict) -> dict:
    """Fill and normalize six measured tables with the common linear field map."""
    xs, ys, mapping, mask = rectangular_mapping(native["positions_arcsec"])
    b = native["strengths"]
    result = {}
    for name, (left, right, _) in FUNCTIONS.items():
        values = native[name][
            mapping.reshape(-1)[:, None], mapping.reshape(-1)[None, :]
        ]
        if left:
            values *= b.reshape(1, 1, len(b), 1, 1, 1)
        if right:
            shape = [1] * values.ndim
            shape[3 if left else 2] = len(b)
            values *= b.reshape(shape)
        result[name] = values.reshape(
            len(xs), len(ys), len(xs), len(ys), *values.shape[2:]
        )
    return {
        **result,
        "x_arcsec": xs,
        "y_arcsec": ys,
        "strengths": b,
        "measured_mask": mask,
        "source_mapping": mapping,
    }
