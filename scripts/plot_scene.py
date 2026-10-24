"""Plot a C13 scene with the cold-cloud threshold contour and print the cold fraction."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, help="a C13 CMIPC .nc file")
    parser.add_argument("--threshold", type=float, default=235.0)
    parser.add_argument("--out", type=Path, default=Path("data/figures/c13.png"))
    args = parser.parse_args()

    with xr.open_dataset(args.path, engine="h5netcdf") as ds:
        bt = ds["CMI"].values
    cold = bt < args.threshold
    print(f"Cold fraction (BT < {args.threshold:.0f} K): {np.nanmean(cold):.2%}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 7))
    image = ax.imshow(bt, cmap="gray_r", vmin=200, vmax=300)
    ax.contour(bt, levels=[args.threshold], colors="red", linewidths=0.6)
    fig.colorbar(image, ax=ax, label="BT 10.3 um (K)", shrink=0.7)
    ax.set_title(f"{args.path.name[:40]} - red: {args.threshold:.0f} K")
    fig.savefig(args.out, dpi=100, bbox_inches="tight")
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
