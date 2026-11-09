"""Check that a freshly ingested store matches the same times of an existing store."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import xarray as xr


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("new", type=Path, help="the store written by the new download path")
    parser.add_argument("old", type=Path, help="the existing store to compare with")
    args = parser.parse_args()

    new = xr.open_zarr(args.new)
    old = xr.open_zarr(args.old).sel(time=new["time"], method=None)
    print(f"times: new {new.sizes['time']}, matched in old {old.sizes['time']}")
    same = np.array_equal(new["bt"].values, old["bt"].values, equal_nan=True)
    print(f"values identical: {same}")
    print(f"corners identical: {new.attrs['corners_latlon'] == old.attrs['corners_latlon']}")
    if not same:
        raise SystemExit("The stores differ.")


if __name__ == "__main__":
    main()
