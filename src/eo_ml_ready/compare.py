"""Compare trained CNN seeds with persistence and advection on the test days (paired intervals)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import xarray as xr

from eo_ml_ready.baseline import METHODS, evaluate_method
from eo_ml_ready.metrics import block_bootstrap, block_bootstrap_diff, scores
from eo_ml_ready.model import SmallUNet
from eo_ml_ready.samples import block_ids, sample_triplets, split_triplets
from eo_ml_ready.train import THRESHOLDS, predict_counts


def interval(pair: tuple[float, float]) -> str:
    return f"{pair[0]:+.3f} to {pair[1]:+.3f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument("--runs", type=Path, required=True, help="folder holding seed*/ results")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--val-start", default="2026-09-07")
    parser.add_argument("--test-start", default="2026-09-09")
    parser.add_argument("--out", type=Path, default=None, help="markdown table to write")
    args = parser.parse_args()

    ds = xr.open_zarr(args.store)
    times, bt = ds["time"].values, ds["bt"]
    splits = split_triplets(times, sample_triplets(times), args.val_start, args.test_start)
    part = splits["test"]
    blocks = block_ids(times, part)

    counts = {name: evaluate_method(bt, part, predict) for name, predict in METHODS.items()}
    rows = [
        "| Method | POD | FAR | CSI (95% CI) | CSI minus advection (paired 95% CI) |",
        "|---|---|---|---|---|",
    ]
    for name in ("persistence", "advection"):
        s, ci = scores(counts[name].sum(axis=0)), block_bootstrap(counts[name], blocks)["csi"]
        diff = block_bootstrap_diff(counts[name], counts["advection"], blocks)["csi"]
        rows.append(
            f"| {name.capitalize()} | {s['pod']:.3f} | {s['far']:.3f} | "
            f"{s['csi']:.3f} ({ci[0]:.3f}-{ci[1]:.3f}) | "
            f"{'-' if name == 'advection' else interval(diff)} |"
        )

    for seed in args.seeds:
        run = args.runs / f"seed{seed}"
        threshold = json.loads((run / "cnn_results.json").read_text())["threshold"]
        model = SmallUNet()
        model.load_state_dict(torch.load(run / "cnn.pt", map_location="cpu", weights_only=True))
        index = int(np.argmin(np.abs(THRESHOLDS - threshold)))
        cnn = predict_counts(model, bt, part)[:, index]
        s, ci = scores(cnn.sum(axis=0)), block_bootstrap(cnn, blocks)["csi"]
        diff = block_bootstrap_diff(cnn, counts["advection"], blocks)["csi"]
        rows.append(
            f"| CNN seed {seed} | {s['pod']:.3f} | {s['far']:.3f} | "
            f"{s['csi']:.3f} ({ci[0]:.3f}-{ci[1]:.3f}) | {interval(diff)} |"
        )
    text = "\n".join(rows)
    print(text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n")


if __name__ == "__main__":
    main()
