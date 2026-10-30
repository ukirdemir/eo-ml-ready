"""Print the metadata of a Zarr store and plot two channels of one frame (orientation check)."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", type=Path, default=Path("data/processed/seviri.zarr"))
    parser.add_argument("--time", default=None, help="e.g. 2026-09-01T15:00 (default: middle)")
    parser.add_argument("--out", type=Path, default=Path("docs/figures/store_frame.png"))
    args = parser.parse_args()

    with xr.open_dataset(args.store, engine="zarr", chunks=None) as ds:
        print(f"Times: {ds.sizes['time']} ({ds['time'].values[0]} to {ds['time'].values[-1]})")
        print(f"Shape: {dict(ds['bt'].sizes)}")
        for key, value in ds.attrs.items():
            print(f"{key}: {value}")
        index = (
            int(np.argmin(np.abs(ds["time"].values - np.datetime64(args.time))))
            if args.time
            else ds.sizes["time"] // 2
        )
        frame = ds["bt"].isel(time=index).astype(np.float32).values
        stamp = str(ds["time"].values[index])[:16]
        channels = [str(c) for c in ds["channel"].values]

    fig, axes = plt.subplots(1, len(channels), figsize=(5 * len(channels), 4.6))
    for ax, name, field in zip(np.atleast_1d(axes), channels, frame, strict=True):
        image = ax.imshow(field, cmap="gray_r")
        ax.set_title(f"{name}, {stamp} UTC (top = north)")
        fig.colorbar(image, ax=ax, label="K", shrink=0.8)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.out, dpi=120)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
