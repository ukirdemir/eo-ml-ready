"""Read MSG SEVIRI native products (Data Store zip files) into north-up cropped brightness
temperature, for the ingest step. Needs the ``eumetsat`` extra (satpy)."""

from __future__ import annotations

import gc
import tempfile
import zipfile
from pathlib import Path

import numpy as np

from eo_ml_ready.ingest import Crop, Region, Source

CHANNELS = ("IR_108", "WV_062")

# North-up pixel window of the 3712 x 3712 full disk (pixel 1856, 1856 is near 0N, 0E).
# About 26N to 3S and 12W to 15E: Sahel, Gulf of Guinea and northern Central Africa.
SEVIRI_REGION = Region(row0=960, row1=1984, col0=1400, col1=2424)


def orient_north_up(
    values: np.ndarray, x: np.ndarray, y: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool, bool]:
    """Flip an array so that rows run north to south and columns west to east.

    SEVIRI native files are stored in scan order (south to north, east to west). The flags tell
    which axes were flipped, so that pixel indices can be mapped back to the original grid.
    """
    flip_cols = bool(x[0] > x[-1])
    flip_rows = bool(y[0] < y[-1])
    if flip_cols:
        values, x = values[:, ::-1], x[::-1]
    if flip_rows:
        values, y = values[::-1], y[::-1]
    return values, x, y, flip_rows, flip_cols


def crop_window(values: np.ndarray, x: np.ndarray, y: np.ndarray, region: Region):
    """Cut the region out of north-up values and coordinates."""
    rows = slice(region.row0, region.row1)
    cols = slice(region.col0, region.col1)
    return values[rows, cols], x[cols], y[rows]


def extract_native(path: Path, workdir: Path) -> Path:
    """Extract the .nat file from a Data Store zip."""
    with zipfile.ZipFile(path) as archive:
        archive.extractall(workdir)
    nat = sorted(workdir.rglob("*.nat"))
    if not nat:
        raise FileNotFoundError(f"No .nat file inside {path}")
    return nat[0]


def corner_latlon(area, region: Region, shape: tuple[int, int], flips: tuple[bool, bool]) -> dict:
    """Latitude and longitude of the four corner pixels of the north-up region."""
    height, width = shape
    flip_rows, flip_cols = flips
    pixels = {
        "nw": (region.row0, region.col0),
        "ne": (region.row0, region.col1 - 1),
        "sw": (region.row1 - 1, region.col0),
        "se": (region.row1 - 1, region.col1 - 1),
    }
    result = {}
    for name, (row, col) in pixels.items():
        source_row = height - 1 - row if flip_rows else row
        source_col = width - 1 - col if flip_cols else col
        lon, lat = area.get_lonlat(source_row, source_col)
        result[name] = [round(float(lat), 3), round(float(lon), 3)]
    return result


def read_seviri(paths: list[Path], region: Region) -> list[Crop]:
    """Read IR 10.8 and WV 6.2 brightness temperature from one product zip."""
    from satpy import Scene

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        nat = extract_native(paths[0], Path(tmp))
        scene = Scene(filenames=[str(nat)], reader="seviri_l1b_native")
        scene.load(list(CHANNELS))
        area = scene[CHANNELS[0]].attrs["area"]
        x_all, y_all = (np.asarray(v, dtype=np.float64) for v in area.get_proj_vectors())
        crops, corners = [], None
        for name in CHANNELS:
            raw = np.asarray(scene[name].values, dtype=np.float32)
            # zero radiances give inf or nan in the calibration; keep only plausible values
            raw = np.where((raw > 100.0) & (raw < 350.0), raw, np.nan).astype(np.float32)
            values, x, y, flip_rows, flip_cols = orient_north_up(raw, x_all, y_all)
            values, x, y = crop_window(values, x, y, region)
            if corners is None:
                corners = corner_latlon(area, region, raw.shape, (flip_rows, flip_cols))
            crops.append(
                Crop(
                    values=np.ascontiguousarray(values),
                    y=np.ascontiguousarray(y),
                    x=np.ascontiguousarray(x),
                    projection={},
                    corners=corners,
                )
            )
        del scene, area
        gc.collect()
    return crops


SEVIRI = Source(
    name="seviri",
    channels=CHANNELS,
    scene_channels=("all",),
    read=read_seviri,
    title="Cropped MSG SEVIRI brightness temperature, 15-minute steps",
    origin="EUMETSAT MSG SEVIRI Level 1.5 (HRSEVIRI)",
    default_region=SEVIRI_REGION,
)
