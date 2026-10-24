"""Build a cropped, time-regular Zarr dataset of brightness temperatures from a provider."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import NamedTuple

import numpy as np
import xarray as xr

from eo_ml_ready.fetch.base import Provider, SceneRef
from eo_ml_ready.fetch.goes_aws import GoesAws
from eo_ml_ready.geo import scan_to_latlon

CHANNELS = ("C13", "C08")
STEP = timedelta(minutes=15)
TOLERANCE = timedelta(minutes=3)


@dataclass(frozen=True)
class Region:
    """A pixel window of the source grid."""

    row0: int
    row1: int
    col0: int
    col1: int


DEFAULT_REGION = Region(row0=476, row1=1500, col0=1400, col1=2424)


class Crop(NamedTuple):
    """Cropped brightness temperature with its scan-angle coordinates."""

    values: np.ndarray
    y: np.ndarray
    x: np.ndarray
    projection: dict


def target_times(start: datetime, end: datetime, step: timedelta = STEP) -> list[datetime]:
    """Return regular times in [start, end)."""
    times = []
    current = start
    while current < end:
        times.append(current)
        current += step
    return times


def match_scenes(
    scenes: list[SceneRef], targets: list[datetime], tolerance: timedelta = TOLERANCE
) -> dict[datetime, SceneRef]:
    """Map each target time to the nearest scene within the tolerance."""
    matched = {}
    for target in targets:
        best = min(scenes, key=lambda ref: abs(ref.time - target), default=None)
        if best is not None and abs(best.time - target) <= tolerance:
            matched[target] = best
    return matched


def read_crop(path: Path, region: Region) -> Crop:
    """Read the CMI field of one file inside the region."""
    window = {"y": slice(region.row0, region.row1), "x": slice(region.col0, region.col1)}
    with xr.open_dataset(path, engine="h5netcdf") as ds:
        cmi = ds["CMI"].isel(window)
        projection = dict(ds["goes_imager_projection"].attrs)
        return Crop(
            values=cmi.values.astype(np.float32),
            y=cmi["y"].values.copy(),
            x=cmi["x"].values.copy(),
            projection=projection,
        )


def build_day(
    provider: Provider,
    start: datetime,
    end: datetime,
    region: Region,
    raw_dir: Path,
    workers: int = 8,
) -> tuple[xr.Dataset, dict] | None:
    """Download, crop and stack all complete time steps in [start, end)."""
    targets = target_times(start, end)
    matched = {
        channel: match_scenes(
            provider.list_scenes(start - TOLERANCE, end + TOLERANCE, channel), targets
        )
        for channel in CHANNELS
    }
    times = [t for t in targets if all(t in matched[c] for c in CHANNELS)]
    if not times:
        return None
    refs = [matched[c][t] for t in times for c in CHANNELS]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        paths = list(pool.map(lambda ref: provider.download(ref, raw_dir), refs))
    crops = []
    for path in paths:
        crops.append(read_crop(path, region))
        path.unlink()
    values = np.stack([crop.values for crop in crops]).astype(np.float16)
    values = values.reshape(len(times), len(CHANNELS), *values.shape[1:])
    dataset = xr.Dataset(
        {"bt": (("time", "channel", "y", "x"), values)},
        coords={
            "time": np.array([t.replace(tzinfo=None) for t in times], dtype="datetime64[ns]"),
            "channel": list(CHANNELS),
            "y": crops[0].y,
            "x": crops[0].x,
        },
    )
    return dataset, crops[0].projection


def stored_times(out: Path) -> np.ndarray:
    """Return the times already written to the store."""
    if not out.exists():
        return np.array([], dtype="datetime64[ns]")
    with xr.open_dataset(out, engine="zarr", chunks=None) as ds:
        return ds["time"].values


def corner_metadata(dataset: xr.Dataset, projection: dict) -> dict:
    """Return lat/lon of the four corners of the cropped grid."""
    y, x = dataset["y"].values, dataset["x"].values
    corners = {"nw": (y[0], x[0]), "ne": (y[0], x[-1]), "sw": (y[-1], x[0]), "se": (y[-1], x[-1])}
    result = {}
    for name, (cy, cx) in corners.items():
        lat, lon = scan_to_latlon(cx, cy, projection)
        result[name] = [round(float(lat), 3), round(float(lon), 3)]
    return result


def ingest(
    provider: Provider,
    start: datetime,
    end: datetime,
    out: Path,
    region: Region = DEFAULT_REGION,
    raw_dir: Path = Path("data/raw/goes"),
    workers: int = 8,
) -> None:
    """Write [start, end) day by day into a Zarr store, skipping days already stored."""
    day = start
    while day < end:
        day_end = min(day + timedelta(days=1), end)
        existing = stored_times(out)
        lo = np.datetime64(day.replace(tzinfo=None))
        hi = np.datetime64(day_end.replace(tzinfo=None))
        if ((existing >= lo) & (existing < hi)).any():
            print(f"{day:%Y-%m-%d}: already stored, skipped")
            day = day_end
            continue
        built = build_day(provider, day, day_end, region, raw_dir, workers)
        if built is None:
            print(f"{day:%Y-%m-%d}: no complete time steps")
            day = day_end
            continue
        dataset, projection = built
        dataset.attrs = {
            "title": "Cropped GOES ABI brightness temperature, 15-minute steps",
            "source": f"NOAA GOES-R ABI L2 CMIPC via {provider.name}",
            "channels": list(CHANNELS),
            "units": "K",
            "region_pixels": [region.row0, region.row1, region.col0, region.col1],
            "corners_latlon": corner_metadata(dataset, projection),
        }
        if out.exists():
            dataset.to_zarr(out, mode="a", append_dim="time")
        else:
            encoding = {"bt": {"chunks": (1, len(CHANNELS), 256, 256)}}
            dataset.to_zarr(out, mode="w", encoding=encoding)
        print(f"{day:%Y-%m-%d}: wrote {dataset.sizes['time']} time steps")
        day = day_end


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", required=True, help="UTC date, e.g. 2026-09-01")
    parser.add_argument("--end", required=True, help="UTC date (exclusive), e.g. 2026-09-11")
    parser.add_argument("--out", type=Path, default=Path("data/processed/goes.zarr"))
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw/goes"))
    parser.add_argument("--satellite", type=int, default=19)
    args = parser.parse_args()
    ingest(
        GoesAws(satellite=args.satellite),
        datetime.fromisoformat(args.start).replace(tzinfo=UTC),
        datetime.fromisoformat(args.end).replace(tzinfo=UTC),
        args.out,
        raw_dir=args.raw_dir,
    )


if __name__ == "__main__":
    main()
