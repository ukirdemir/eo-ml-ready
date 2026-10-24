"""Tests for geolocation, scene matching and the end-to-end ingest with a fake provider."""

from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from eo_ml_ready.fetch.base import SceneRef
from eo_ml_ready.geo import scan_to_latlon
from eo_ml_ready.ingest import CHANNELS, Region, ingest, match_scenes, target_times

PROJECTION = {
    "semi_major_axis": 6378137.0,
    "semi_minor_axis": 6356752.31414,
    "perspective_point_height": 35786023.0,
    "longitude_of_projection_origin": -75.0,
}


def utc(hour, minute, second=0):
    return datetime(2026, 9, 1, hour, minute, second, tzinfo=UTC)


class FakeProvider:
    """Writes tiny CMI files at 15-minute steps with a few seconds of jitter."""

    name = "fake"

    def __init__(self):
        self.downloads = 0

    def list_scenes(self, start, end, channel):
        refs = []
        time = utc(0, 0)
        while time < utc(2, 0):
            sensed = time + timedelta(seconds=17)
            if start <= sensed < end:
                refs.append(SceneRef("fake", f"{channel}_{sensed:%H%M%S}.nc", sensed, channel))
            time += timedelta(minutes=15)
        return refs

    def download(self, ref, dest):
        self.downloads += 1
        dest.mkdir(parents=True, exist_ok=True)
        path = dest / ref.key
        level = 200.0 + ref.time.minute + (10.0 if ref.channel == CHANNELS[1] else 0.0)
        ds = xr.Dataset(
            {
                "CMI": (("y", "x"), np.full((8, 8), level, dtype=np.float32)),
                "goes_imager_projection": ((), 0, PROJECTION),
            },
            coords={"y": np.linspace(0.01, -0.01, 8), "x": np.linspace(-0.01, 0.01, 8)},
        )
        ds.to_netcdf(path, engine="h5netcdf")
        return path


def test_scan_to_latlon_nadir():
    lat, lon = scan_to_latlon(0.0, 0.0, PROJECTION)
    assert lat == pytest.approx(0.0)
    assert lon == pytest.approx(-75.0)


def test_scan_to_latlon_off_disk_is_nan():
    lat, _ = scan_to_latlon(0.5, 0.5, PROJECTION)
    assert np.isnan(lat)


def test_match_scenes_uses_tolerance():
    targets = target_times(utc(0, 0), utc(1, 0))
    scenes = [
        SceneRef("fake", "a", utc(0, 0, 20), "C13"),
        SceneRef("fake", "b", utc(0, 30, 0), "C13"),
    ]
    matched = match_scenes(scenes, targets)
    assert set(matched) == {utc(0, 0), utc(0, 30)}


def test_ingest_writes_and_resumes(tmp_path: Path):
    provider = FakeProvider()
    out = tmp_path / "out.zarr"
    raw = tmp_path / "raw"
    region = Region(row0=2, row1=6, col0=1, col1=5)
    ingest(provider, utc(0, 0), utc(1, 0), out, region=region, raw_dir=raw, workers=2)
    with xr.open_dataset(out, engine="zarr", chunks=None) as ds:
        assert ds["bt"].shape == (4, 2, 4, 4)
        assert ds["bt"].dtype == np.float16
        assert ds.attrs["channels"] == list(CHANNELS)
        assert float(ds["bt"].isel(time=1, channel=0).mean()) == pytest.approx(215.0)
    assert not list(raw.glob("*.nc"))
    first_run = provider.downloads
    ingest(provider, utc(0, 0), utc(1, 0), out, region=region, raw_dir=raw, workers=2)
    assert provider.downloads == first_run
