"""Download one GOES ABI CONUS scene per channel and print grid, units and value ranges."""

from __future__ import annotations

import argparse
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import xarray as xr

from eo_ml_ready.fetch.goes_aws import GoesAws

CHANNELS = ["C13", "C08"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--time", default="2026-09-15T18:00", help="UTC start, ISO format")
    parser.add_argument("--satellite", type=int, default=19)
    parser.add_argument("--dest", type=Path, default=Path("data/raw/goes"))
    args = parser.parse_args()

    start = datetime.fromisoformat(args.time).replace(tzinfo=UTC)
    provider = GoesAws(satellite=args.satellite)
    for channel in CHANNELS:
        scenes = provider.list_scenes(start, start + timedelta(hours=1), channel)
        print(f"{channel}: {len(scenes)} scenes in the hour")
        if not scenes:
            continue
        begin = time.perf_counter()
        path = provider.download(scenes[0], args.dest)
        with xr.open_dataset(path, engine="h5netcdf") as ds:
            cmi = ds["CMI"]
            values = cmi.values
            print(f"  file {path.name}, {path.stat().st_size / 1e6:.1f} MB")
            print(f"  shape {values.shape}, units {cmi.attrs.get('units')}")
            print(f"  valid {np.isfinite(values).mean():.1%}")
            print(f"  min {np.nanmin(values):.1f}, max {np.nanmax(values):.1f}")
            projection = ds["goes_imager_projection"].attrs
            print(f"  satellite height {projection.get('perspective_point_height')} m")
        print(f"  time {time.perf_counter() - begin:.1f} s")


if __name__ == "__main__":
    main()
