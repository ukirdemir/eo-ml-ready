"""Check a brightness-temperature Zarr store: infinities, missing pixels, ranges, cold fraction."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import xarray as xr

THRESHOLD_K = 235.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", type=Path, default=Path("data/processed/seviri.zarr"))
    parser.add_argument("--nan-limit", type=float, default=0.01, help="flag frames above this")
    args = parser.parse_args()

    with xr.open_dataset(args.store, engine="zarr", chunks=None) as ds:
        times = ds["time"].values
        channels = [str(c) for c in ds["channel"].values]
        print(f"{len(times)} frames, channels {channels}")
        flagged = []
        cold = []
        change = []
        previous = None
        stats = {name: [np.inf, -np.inf, 0, 0] for name in channels}
        for index in range(len(times)):
            frame = ds["bt"].isel(time=index).astype(np.float32).values
            for k, name in enumerate(channels):
                field = frame[k]
                stats[name][2] += int(np.isinf(field).sum())
                stats[name][3] += int(np.isnan(field).sum())
                finite = field[np.isfinite(field)]
                if finite.size:
                    stats[name][0] = min(stats[name][0], float(finite.min()))
                    stats[name][1] = max(stats[name][1], float(finite.max()))
            if previous is not None:
                change.append(float(np.nanmean(np.abs(frame[0] - previous))))
            previous = frame[0]
            bad = ~np.isfinite(frame[0]) | ~np.isfinite(frame[1])
            if bad.mean() > args.nan_limit:
                flagged.append((str(times[index])[:16], float(bad.mean())))
            with np.errstate(invalid="ignore"):
                cold.append(float(np.nanmean(frame[0] < THRESHOLD_K)))
        for name, (low, high, n_inf, n_nan) in stats.items():
            print(f"{name}: min {low:.1f} K, max {high:.1f} K, inf {n_inf}, nan {n_nan}")
        print(
            f"Cold fraction (BT < {THRESHOLD_K:.0f} K): mean {np.mean(cold):.2%}, "
            f"max {np.max(cold):.2%}"
        )
        same = int(np.sum(np.array(change) < 0.01))
        print(
            f"Mean |change| of {channels[0]} between consecutive frames: "
            f"{np.mean(change):.2f} K; near-identical neighbours: {same}"
        )
        print(f"Frames with more than {args.nan_limit:.0%} missing: {len(flagged)}")
        for stamp, share in flagged:
            print(f"  {stamp}  {share:.1%} missing")


if __name__ == "__main__":
    main()
