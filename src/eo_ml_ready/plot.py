"""Plot observation, persistence and CNN forecasts for one test sample."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import xarray as xr
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

from eo_ml_ready.model import SmallUNet
from eo_ml_ready.samples import (
    THRESHOLD_K,
    cold_mask,
    normalise,
    read_sample,
    sample_triplets,
    split_triplets,
)

CODES = ListedColormap(["#f2f2f2", "#2ca02c", "#1f77b4", "#d62728"])
LEGEND = [
    Patch(color="#2ca02c", label="hit"),
    Patch(color="#1f77b4", label="miss"),
    Patch(color="#d62728", label="false alarm"),
]


def categorical(pred: np.ndarray, obs: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, float]:
    """Return a code map (0 none, 1 hit, 2 miss, 3 false alarm; NaN invalid) and the frame CSI."""
    codes = np.zeros(pred.shape, dtype=float)
    codes[pred & obs] = 1
    codes[~pred & obs] = 2
    codes[pred & ~obs] = 3
    codes[~valid] = np.nan
    hits, misses, alarms = (np.sum(codes == c) for c in (1, 2, 3))
    total = hits + misses + alarms
    return codes, float(hits / total) if total else float("nan")


def pick_sample(bt: xr.DataArray, times: np.ndarray, triplets: np.ndarray, when: str | None):
    """Return the triplet at the given origin time, or the one with the most cold cloud."""
    if when:
        match = np.flatnonzero(times[triplets[:, 1]] == np.datetime64(when))
        if len(match) == 0:
            raise SystemExit(f"No sample with origin time {when} in this split")
        return triplets[match[0]]
    fractions = [np.nanmean(cold_mask(read_sample(bt, t)[1])) for t in triplets]
    return triplets[int(np.argmax(fractions))]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", type=Path, default=Path("data/processed/goes.zarr"))
    parser.add_argument("--run", type=Path, default=Path("artifacts/seed0"))
    parser.add_argument("--split", default="test", choices=["val", "test"])
    parser.add_argument("--time", default=None, help="origin time, e.g. 2026-09-09T18:00")
    parser.add_argument("--val-start", default="2026-09-07")
    parser.add_argument("--test-start", default="2026-09-09")
    parser.add_argument("--out", type=Path, default=Path("docs/figures/forecast.png"))
    args = parser.parse_args()

    threshold = json.loads((args.run / "cnn_results.json").read_text())["threshold"]
    model = SmallUNet()
    model.load_state_dict(torch.load(args.run / "cnn.pt", map_location="cpu"))
    model.eval()

    ds = xr.open_zarr(args.store)
    times, bt = ds["time"].values, ds["bt"]
    splits = split_triplets(times, sample_triplets(times), args.val_start, args.test_start)
    triplet = pick_sample(bt, times, splits[args.split], args.time)
    inputs, target = read_sample(bt, triplet)
    with torch.no_grad():
        prob = torch.sigmoid(model(torch.from_numpy(normalise(inputs))[None]))[0, 0].numpy()

    obs, valid = cold_mask(target), np.isfinite(target)
    persistence, cnn = cold_mask(inputs[2]), prob >= threshold
    persistence_map, persistence_csi = categorical(persistence, obs, valid)
    cnn_map, cnn_csi = categorical(cnn, obs, valid)
    origin, lead = (str(times[i])[:16] for i in triplet[1:])

    fig, axes = plt.subplots(2, 2, figsize=(12, 11))
    panels = [
        (axes[0, 0], inputs[2], f"Observed BT 10.3 um at t = {origin}"),
        (axes[0, 1], target, f"Observed BT 10.3 um at t+60 = {lead}"),
    ]
    for ax, field, title in panels:
        ax.imshow(field, cmap="gray_r", vmin=200, vmax=300)
        ax.contour(field, levels=[THRESHOLD_K], colors="red", linewidths=0.6)
        ax.set_title(title, fontsize=10)
    for ax, codes, title in [
        (axes[1, 0], persistence_map, f"Persistence vs observed, CSI {persistence_csi:.2f}"),
        (axes[1, 1], cnn_map, f"CNN vs observed, CSI {cnn_csi:.2f} (threshold {threshold})"),
    ]:
        ax.imshow(codes, cmap=CODES, vmin=0, vmax=3, interpolation="nearest")
        ax.set_title(title, fontsize=10)
        ax.legend(handles=LEGEND, loc="lower left", fontsize=8)
    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("Cold-cloud (BT < 235 K) nowcast, +60 min", fontsize=12)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=100, bbox_inches="tight")
    print(f"Saved {args.out} (origin {origin}, frame CSI persistence {persistence_csi:.2f}, "
          f"CNN {cnn_csi:.2f})")


if __name__ == "__main__":
    main()
