"""Persistence baseline: the cold-cloud mask at t is the forecast for t+60."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import xarray as xr

from eo_ml_ready.metrics import block_bootstrap, count_events, scores
from eo_ml_ready.samples import (
    block_ids,
    cold_mask,
    read_sample,
    sample_triplets,
    split_triplets,
)


def evaluate_persistence(bt: xr.DataArray, triplets: np.ndarray) -> np.ndarray:
    """Return per-sample event counts for the persistence forecast."""
    counts = []
    for triplet in triplets:
        inputs, target = read_sample(bt, triplet)
        pred = cold_mask(inputs[2])
        counts.append(count_events(pred, cold_mask(target), np.isfinite(target)))
    return np.stack(counts)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", type=Path, default=Path("data/processed/goes.zarr"))
    parser.add_argument("--val-start", default="2026-09-07")
    parser.add_argument("--test-start", default="2026-09-09")
    parser.add_argument("--out", type=Path, default=Path("artifacts/persistence.json"))
    args = parser.parse_args()

    ds = xr.open_zarr(args.store)
    times = ds["time"].values
    triplets = sample_triplets(times)
    splits = split_triplets(times, triplets, args.val_start, args.test_start)
    bt = ds["bt"]
    results = {}
    for name in ("val", "test"):
        part = splits[name]
        counts = evaluate_persistence(bt, part)
        total = counts.sum(axis=0)
        results[name] = {
            "samples": len(part),
            "cold_fraction": float((total[0] + total[1]) / total.sum()),
            "scores": scores(total),
            "ci95": block_bootstrap(counts, block_ids(times, part)),
        }
        print(name, json.dumps(results[name], indent=2))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
