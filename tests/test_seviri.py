"""Tests for the SEVIRI orientation and crop helpers and for single-file multi-channel ingest."""

from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from eo_ml_ready.fetch.base import SceneRef
from eo_ml_ready.ingest import Crop, Region, Source, ingest
from eo_ml_ready.seviri import corner_latlon, crop_window, orient_north_up


def test_orient_flips_scan_order_to_north_up():
    values = np.arange(12, dtype=np.float32).reshape(3, 4)
    x = np.array([30.0, 20.0, 10.0, 0.0])  # east to west
    y = np.array([-10.0, 0.0, 10.0])  # south to north
    out, out_x, out_y, flip_rows, flip_cols = orient_north_up(values, x, y)
    assert flip_rows and flip_cols
    assert list(out_x) == [0.0, 10.0, 20.0, 30.0]
    assert list(out_y) == [10.0, 0.0, -10.0]
    assert out[0, 0] == values[2, 3]
    assert out[2, 3] == values[0, 0]


def test_orient_keeps_north_up_arrays():
    values = np.arange(6, dtype=np.float32).reshape(2, 3)
    x = np.array([0.0, 1.0, 2.0])
    y = np.array([1.0, 0.0])
    out, _, _, flip_rows, flip_cols = orient_north_up(values, x, y)
    assert not flip_rows and not flip_cols
    assert np.array_equal(out, values)


def test_crop_window_cuts_values_and_coordinates():
    values = np.arange(20, dtype=np.float32).reshape(4, 5)
    x = np.arange(5.0)
    y = np.arange(4.0)
    out, out_x, out_y = crop_window(values, x, y, Region(row0=1, row1=3, col0=2, col1=5))
    assert out.shape == (2, 3)
    assert list(out_x) == [2.0, 3.0, 4.0]
    assert list(out_y) == [1.0, 2.0]


class FakeArea:
    """Returns the source pixel indices as (lon, lat) so the index mapping can be checked."""

    def get_lonlat(self, row, col):
        return float(col), float(row)


def test_corner_latlon_maps_back_to_source_indices():
    region = Region(row0=1, row1=3, col0=2, col1=4)
    corners = corner_latlon(FakeArea(), region, (10, 20), (True, True))
    # north-up pixel (1, 2) is source pixel (10 - 1 - 1, 20 - 1 - 2)
    assert corners["nw"] == [8.0, 17.0]
    assert corners["se"] == [7.0, 16.0]


class OneFileProvider:
    """One file per time step holding both channels, as for SEVIRI."""

    name = "fake"

    def __init__(self):
        self.downloads = 0

    def list_scenes(self, start, end, channel):
        refs = []
        time = datetime(2026, 9, 1, 0, 0, tzinfo=UTC)
        while time < datetime(2026, 9, 1, 2, 0, tzinfo=UTC):
            if start <= time < end:
                refs.append(SceneRef("fake", f"{time:%H%M}.nc", time, "all"))
            time += timedelta(minutes=15)
        return refs

    def download(self, ref, dest):
        self.downloads += 1
        dest.mkdir(parents=True, exist_ok=True)
        path = dest / ref.key
        xr.Dataset({"level": ((), float(ref.time.minute))}).to_netcdf(path, engine="h5netcdf")
        return path


def read_two_channels(paths: list[Path], region: Region) -> list[Crop]:
    with xr.open_dataset(paths[0], engine="h5netcdf") as ds:
        level = float(ds["level"])
    shape = (region.row1 - region.row0, region.col1 - region.col0)
    y, x = np.arange(shape[0], dtype=float), np.arange(shape[1], dtype=float)
    corners = {"nw": [1.0, 2.0], "ne": [1.0, 3.0], "sw": [0.0, 2.0], "se": [0.0, 3.0]}
    return [
        Crop(np.full(shape, 200.0 + level + offset, dtype=np.float32), y, x, {}, corners)
        for offset in (0.0, 10.0)
    ]


def test_single_file_source_writes_both_channels(tmp_path: Path):
    source = Source(
        name="fake",
        channels=("A", "B"),
        scene_channels=("all",),
        read=read_two_channels,
        title="fake",
        origin="fake",
        default_region=Region(0, 4, 0, 4),
    )
    provider = OneFileProvider()
    out = tmp_path / "out.zarr"
    start = datetime(2026, 9, 1, 0, 0, tzinfo=UTC)
    end = start + timedelta(hours=1)
    ingest(provider, start, end, out, raw_dir=tmp_path / "raw", source=source)
    with xr.open_dataset(out, engine="zarr", chunks=None) as ds:
        assert ds["bt"].shape == (4, 2, 4, 4)
        assert list(ds["channel"].values) == ["A", "B"]
        assert float(ds["bt"].isel(time=1, channel=1).mean()) == pytest.approx(225.0)
        assert ds.attrs["corners_latlon"]["nw"] == [1.0, 2.0]
    assert provider.downloads == 4
    assert not list((tmp_path / "raw").glob("*.nc"))
