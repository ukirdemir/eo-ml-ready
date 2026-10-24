"""Convert GOES fixed-grid scan angles to latitude and longitude."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np


def scan_to_latlon(x, y, projection: Mapping) -> tuple[np.ndarray, np.ndarray]:
    """Return (lat, lon) in degrees for scan angles x, y in radians; NaN off the Earth disk."""
    req = float(projection["semi_major_axis"])
    rpol = float(projection["semi_minor_axis"])
    height = float(projection["perspective_point_height"]) + req
    lon0 = np.radians(float(projection["longitude_of_projection_origin"]))
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    a = np.sin(x) ** 2 + np.cos(x) ** 2 * (np.cos(y) ** 2 + (req**2 / rpol**2) * np.sin(y) ** 2)
    b = -2.0 * height * np.cos(x) * np.cos(y)
    c = height**2 - req**2
    with np.errstate(invalid="ignore"):
        rs = (-b - np.sqrt(b**2 - 4.0 * a * c)) / (2.0 * a)
    sx = rs * np.cos(x) * np.cos(y)
    sy = -rs * np.sin(x)
    sz = rs * np.cos(x) * np.sin(y)
    lat = np.arctan((req**2 / rpol**2) * sz / np.sqrt((height - sx) ** 2 + sy**2))
    lon = lon0 - np.arctan(sy / (height - sx))
    return np.degrees(lat), np.degrees(lon)
