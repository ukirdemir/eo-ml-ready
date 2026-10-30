"""Baselines for the cold-cloud nowcast: persistence and optical-flow advection.

Persistence: the cold-cloud mask at t is the forecast for t+60.
Advection: the brightness temperature at t is moved along the optical flow between t-15 and t
(four more steps of the same flow, as a constant-motion assumption) and then thresholded.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from pathlib import Path

import numpy as np
import xarray as xr

from eo_ml_ready.metrics import block_bootstrap, count_events, scores
from eo_ml_ready.samples import (
    LAG,
    LEAD,
    block_ids,
    cold_mask,
    read_sample,
    sample_triplets,
    split_triplets,
)

WARM_K = 300.0  # fill value outside the source field and for missing pixels
FLOW_SIGMA = 8.0


def flow_image(field_k: np.ndarray) -> np.ndarray:
    """Scale brightness temperature to 8 bits; only cold cloud (200-270 K) has structure."""
    cold = np.clip((270.0 - np.nan_to_num(field_k, nan=WARM_K)) / 70.0, 0.0, 1.0)
    return (cold * 255).astype(np.uint8)


def advect(field_k: np.ndarray, flow_per_step: np.ndarray, steps: int) -> np.ndarray:
    """Move a brightness-temperature field along a per-step flow (pixels) for ``steps`` steps."""
    import cv2

    height, width = flow_per_step.shape[:2]
    grid_x, grid_y = np.meshgrid(np.arange(width), np.arange(height))
    map_x = (grid_x - steps * flow_per_step[..., 0]).astype(np.float32)
    map_y = (grid_y - steps * flow_per_step[..., 1]).astype(np.float32)
    source = np.nan_to_num(field_k, nan=WARM_K).astype(np.float32)
    return cv2.remap(
        source, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=WARM_K
    )


def estimate_flow(previous_k: np.ndarray, latest_k: np.ndarray) -> np.ndarray:
    """DIS optical flow between two frames, smoothed where there is cloud (pixels per step)."""
    import cv2

    previous, latest = flow_image(previous_k), flow_image(latest_k)
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    flow = dis.calc(previous, latest, None)
    weight = ((previous > 0) | (latest > 0)).astype(np.float32)
    smooth_num = cv2.GaussianBlur(flow * weight[..., None], (0, 0), FLOW_SIGMA)
    smooth_den = cv2.GaussianBlur(weight, (0, 0), FLOW_SIGMA)[..., None]
    mean_flow = (flow * weight[..., None]).sum(axis=(0, 1)) / max(float(weight.sum()), 1.0)
    return np.where(smooth_den > 1e-3, smooth_num / np.maximum(smooth_den, 1e-3), mean_flow)


def persistence_mask(inputs: np.ndarray) -> np.ndarray:
    """Forecast the latest cold-cloud mask unchanged (inputs: lag ch0, lag ch1, now ch0, ...)."""
    return cold_mask(inputs[2])


def advection_mask(inputs: np.ndarray) -> np.ndarray:
    """Forecast the cold-cloud mask by advecting the latest frame along the optical flow."""
    flow = estimate_flow(inputs[0], inputs[2])
    return cold_mask(advect(inputs[2], flow, steps=int(LEAD / LAG)))


METHODS: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "persistence": persistence_mask,
    "advection": advection_mask,
}


def evaluate_method(
    bt: xr.DataArray, triplets: np.ndarray, predict: Callable[[np.ndarray], np.ndarray]
) -> np.ndarray:
    """Return per-sample event counts for a mask-forecasting method."""
    counts = []
    for triplet in triplets:
        inputs, target = read_sample(bt, triplet)
        counts.append(count_events(predict(inputs), cold_mask(target), np.isfinite(target)))
    return np.stack(counts)


def evaluate_persistence(bt: xr.DataArray, triplets: np.ndarray) -> np.ndarray:
    """Return per-sample event counts for the persistence forecast."""
    return evaluate_method(bt, triplets, persistence_mask)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", type=Path, default=Path("data/processed/goes.zarr"))
    parser.add_argument("--val-start", default="2026-09-07")
    parser.add_argument("--test-start", default="2026-09-09")
    parser.add_argument("--method", choices=sorted(METHODS), default="persistence")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    out = args.out or Path(f"artifacts/{args.method}.json")

    ds = xr.open_zarr(args.store)
    times = ds["time"].values
    triplets = sample_triplets(times)
    splits = split_triplets(times, triplets, args.val_start, args.test_start)
    bt = ds["bt"]
    results = {}
    for name in ("val", "test"):
        part = splits[name]
        counts = evaluate_method(bt, part, METHODS[args.method])
        total = counts.sum(axis=0)
        results[name] = {
            "samples": len(part),
            "cold_fraction": float((total[0] + total[1]) / total.sum()),
            "scores": scores(total),
            "ci95": block_bootstrap(counts, block_ids(times, part)),
        }
        print(name, json.dumps(results[name], indent=2))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
