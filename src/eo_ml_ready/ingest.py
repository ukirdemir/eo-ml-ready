"""Build a cropped, time-regular Zarr dataset of brightness temperatures from a provider."""

from __future__ import annotations

import argparse
from collections.abc import Callable
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
CHUNK_STEPS = 8


@dataclass(frozen=True)
class Region:
    """A pixel window of the source grid."""

    row0: int
    row1: int
    col0: int
    col1: int


DEFAULT_REGION = Region(row0=476, row1=1500, col0=1400, col1=2424)


class Crop(NamedTuple):
    """Cropped brightness temperature with its coordinates (and corner lat/lon if known)."""

    values: np.ndarray
    y: np.ndarray
    x: np.ndarray
    projection: dict
    corners: dict | None = None


@dataclass(frozen=True)
class Source:
    """How the files of one sensor map to the stored channels.

    ``scene_channels`` are the keys used to list and download scenes (GOES has one file per
    channel; SEVIRI has one file holding all channels). ``read`` receives the downloaded paths of
    one time step, in ``scene_channels`` order, and returns one Crop per stored channel.
    """

    name: str
    channels: tuple[str, ...]
    scene_channels: tuple[str, ...]
    read: Callable[[list[Path], Region], list[Crop]]
    title: str
    origin: str
    default_region: Region


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
    """Read the CMI field of one GOES file inside the region."""
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


def read_goes(paths: list[Path], region: Region) -> list[Crop]:
    """Read one GOES file per channel."""
    return [read_crop(path, region) for path in paths]


GOES = Source(
    name="goes",
    channels=CHANNELS,
    scene_channels=CHANNELS,
    read=read_goes,
    title="Cropped GOES ABI brightness temperature, 15-minute steps",
    origin="NOAA GOES-R ABI L2 CMIPC",
    default_region=DEFAULT_REGION,
)


def build_day(
    provider: Provider,
    start: datetime,
    end: datetime,
    region: Region,
    raw_dir: Path,
    workers: int = 8,
    source: Source = GOES,
) -> tuple[xr.Dataset, dict, dict | None] | None:
    """Download, crop and stack all complete time steps in [start, end).

    Time steps are processed in small chunks so that only a few raw files are on disk at once.
    Returns the dataset, the projection and the corner lat/lon (None if the source gives none).
    """
    targets = target_times(start, end)
    matched = {
        key: match_scenes(provider.list_scenes(start - TOLERANCE, end + TOLERANCE, key), targets)
        for key in source.scene_channels
    }
    times = [t for t in targets if all(t in matched[key] for key in source.scene_channels)]
    if not times:
        return None
    per_time = len(source.scene_channels)
    frames: list[np.ndarray] = []
    first: list[Crop] | None = None
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i in range(0, len(times), CHUNK_STEPS):
            chunk = times[i : i + CHUNK_STEPS]
            refs = [matched[key][t] for t in chunk for key in source.scene_channels]
            paths = list(pool.map(lambda ref: provider.download(ref, raw_dir), refs))
            for j in range(len(chunk)):
                group = paths[j * per_time : (j + 1) * per_time]
                crops = source.read(group, region)
                for path in group:
                    path.unlink()
                if first is None:
                    first = crops
                frames.append(np.stack([crop.values for crop in crops]).astype(np.float16))
    assert first is not None
    dataset = xr.Dataset(
        {"bt": (("time", "channel", "y", "x"), np.stack(frames))},
        coords={
            "time": np.array([t.replace(tzinfo=None) for t in times], dtype="datetime64[ns]"),
            "channel": list(source.channels),
            "y": first[0].y,
            "x": first[0].x,
        },
    )
    return dataset, first[0].projection, first[0].corners


def stored_times(out: Path) -> np.ndarray:
    """Return the times already written to the store."""
    if not out.exists():
        return np.array([], dtype="datetime64[ns]")
    with xr.open_dataset(out, engine="zarr", chunks=None) as ds:
        return ds["time"].values


def corner_metadata(dataset: xr.Dataset, projection: dict) -> dict:
    """Return lat/lon of the four corners of the cropped GOES grid."""
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
    region: Region | None = None,
    raw_dir: Path = Path("data/raw/goes"),
    workers: int = 8,
    source: Source = GOES,
) -> None:
    """Write [start, end) day by day into a Zarr store, skipping days already stored."""
    region = region or source.default_region
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
        built = build_day(provider, day, day_end, region, raw_dir, workers, source)
        if built is None:
            print(f"{day:%Y-%m-%d}: no complete time steps")
            day = day_end
            continue
        dataset, projection, corners = built
        dataset.attrs = {
            "title": source.title,
            "source": f"{source.origin} via {provider.name}",
            "channels": list(source.channels),
            "units": "K",
            "region_pixels": [region.row0, region.row1, region.col0, region.col1],
            "corners_latlon": corners or corner_metadata(dataset, projection),
        }
        if out.exists():
            dataset.to_zarr(out, mode="a", append_dim="time")
        else:
            encoding = {"bt": {"chunks": (1, len(source.channels), 256, 256)}}
            dataset.to_zarr(out, mode="w", encoding=encoding)
        print(f"{day:%Y-%m-%d}: wrote {dataset.sizes['time']} time steps")
        day = day_end


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", required=True, help="UTC date, e.g. 2026-09-01")
    parser.add_argument("--end", required=True, help="UTC date (exclusive), e.g. 2026-09-11")
    parser.add_argument("--source", choices=["goes", "seviri"], default="goes")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--raw-dir", type=Path, default=None)
    parser.add_argument("--satellite", type=int, default=19, help="GOES satellite number")
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument(
        "--region",
        type=int,
        nargs=4,
        metavar=("ROW0", "ROW1", "COL0", "COL1"),
        default=None,
        help="pixel window of the source grid (default: the source's own region)",
    )
    args = parser.parse_args()
    start = datetime.fromisoformat(args.start).replace(tzinfo=UTC)
    end = datetime.fromisoformat(args.end).replace(tzinfo=UTC)
    region = Region(*args.region) if args.region else None
    if args.source == "seviri":
        from eo_ml_ready.fetch.eumetsat import Eumetsat
        from eo_ml_ready.seviri import SEVIRI

        provider, source, workers = Eumetsat(), SEVIRI, args.workers or 4
    else:
        provider, source, workers = GoesAws(satellite=args.satellite), GOES, args.workers or 8
    ingest(
        provider,
        start,
        end,
        args.out or Path(f"data/processed/{args.source}.zarr"),
        region=region,
        raw_dir=args.raw_dir or Path(f"data/raw/{args.source}"),
        workers=workers,
        source=source,
    )


if __name__ == "__main__":
    main()
