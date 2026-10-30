"""Load released weights and forecast one frame: cold-cloud mask 60 minutes ahead.

Usage (after downloading the weights and the matching .json from the release):

    python examples/predict.py --weights eo-ml-ready-unet-goes19-seed0.pt
        --store data/processed/goes.zarr --origin 2026-09-09T18:00
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import xarray as xr

from eo_ml_ready.metrics import count_events, scores
from eo_ml_ready.model import SmallUNet
from eo_ml_ready.samples import cold_mask, normalise, read_sample, sample_triplets


def load_model(weights: Path) -> tuple[SmallUNet, dict]:
    """Load the network and the metadata JSON that sits next to the weights."""
    metadata = json.loads(weights.with_suffix(".json").read_text())
    model = SmallUNet()
    model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
    return model.eval(), metadata


def forecast(model: SmallUNet, inputs_k: np.ndarray) -> np.ndarray:
    """Return the probability of cold cloud at t+60 for inputs of shape (4, H, W) in kelvin."""
    with torch.no_grad():
        logits = model(torch.from_numpy(normalise(inputs_k))[None])
    return torch.sigmoid(logits)[0, 0].numpy()


def describe(name: str, s: dict) -> str:
    """Format POD, FAR and CSI on one line."""
    return f"  {name + ':':<13}POD {s['pod']:.2f}  FAR {s['far']:.2f}  CSI {s['csi']:.2f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument("--origin", required=True, help="time t, e.g. 2026-09-09T18:00")
    parser.add_argument("--out", type=Path, default=Path("forecast.png"))
    args = parser.parse_args()

    model, metadata = load_model(args.weights)
    ds = xr.open_zarr(args.store)
    times = ds["time"].values
    triplets = sample_triplets(times)
    wanted = np.datetime64(args.origin)
    match = triplets[times[triplets[:, 1]] == wanted]
    if not len(match):
        raise SystemExit(f"No sample with frames at t-15, t and t+60 for {args.origin}")

    inputs, target = read_sample(ds["bt"], match[0])
    probability = forecast(model, inputs)
    predicted = probability >= metadata["threshold"]
    observed, valid = cold_mask(target), np.isfinite(target)
    result = scores(count_events(predicted, observed, valid))
    persistence = scores(count_events(cold_mask(inputs[2]), observed, valid))
    print(f"{metadata['name']}, origin {args.origin}, threshold {metadata['threshold']}")
    print(describe("model", result))
    print(describe("persistence", persistence))

    import matplotlib.pyplot as plt

    panels = [
        ("Window channel at t (K)", inputs[2], "gray_r"),
        ("Forecast probability", probability, "viridis"),
        ("Observed cold cloud at t+60", observed.astype(float), "Blues"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4))
    for ax, (title, field, cmap) in zip(axes, panels, strict=True):
        image = ax.imshow(field, cmap=cmap)
        ax.set_title(title)
        fig.colorbar(image, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(args.out, dpi=110)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
