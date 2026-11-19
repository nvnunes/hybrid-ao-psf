"""Coordinate conventions used by the Hybrid engine and result files."""

import numpy as np
from astropy import units as u


def _cartesian_to_polar(positions: u.Quantity) -> tuple[u.Quantity, u.Quantity]:
    x_value, y_value = positions.to_value(u.arcsec).T
    radius = np.hypot(x_value, y_value)
    theta = np.mod(np.rad2deg(np.arctan2(y_value, x_value)), 360.0)
    theta[radius == 0.0] = 0.0
    return radius * u.arcsec, theta * u.deg
