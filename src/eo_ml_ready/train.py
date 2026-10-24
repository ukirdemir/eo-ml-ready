"""Train the CNN on random crops, tune its threshold on validation and score it on test."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import xarray as xr
from torch.nn import functional as F

from eo_ml_ready.baseline import evaluate_persistence
from eo_ml_ready.metrics import block_bootstrap, block_bootstrap_diff, count_events, scores
from eo_ml_ready.model import SmallUNet
from eo_ml_ready.samples import (
    block_ids,
    cold_mask,
    normalise,
    read_sample,
    sample_triplets,
    split_triplets,
)

THRESHOLDS = np.round(np.arange(0.05, 0.96, 0.05), 2)


def make_batch(
    frames: np.ndarray, triplets: np.ndarray, rng: np.random.Generator, batch: int, crop: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return random crops: inputs (B, 4, h, w), cold target and valid mask (B, 1, h, w)."""
    height, width = frames.shape[-2:]
    inputs, targets, valid = [], [], []
    for index in rng.integers(0, len(triplets), batch):
        lag, now, lead = triplets[index]
        row = int(rng.integers(0, height - crop + 1))
        col = int(rng.integers(0, width - crop + 1))
        window = (slice(None), slice(row, row + crop), slice(col, col + crop))
        x = np.concatenate([frames[lag][window], frames[now][window]]).astype(np.float32)
        target = frames[lead][0][window[1:]].astype(np.float32)
        inputs.append(normalise(x))
        targets.append(cold_mask(target).astype(np.float32))
        valid.append(np.isfinite(target).astype(np.float32))
    return (
        torch.from_numpy(np.stack(inputs)),
        torch.from_numpy(np.stack(targets))[:, None],
        torch.from_numpy(np.stack(valid))[:, None],
    )


def predict_counts(model: SmallUNet, bt: xr.DataArray, triplets: np.ndarray) -> np.ndarray:
    """Return event counts per sample and threshold, shape (n, thresholds, 4)."""
    model.eval()
    out = []
    with torch.no_grad():
        for triplet in triplets:
            inputs, target = read_sample(bt, triplet)
            x = torch.from_numpy(normalise(inputs))[None]
            prob = torch.sigmoid(model(x))[0, 0].numpy()
            obs, valid = cold_mask(target), np.isfinite(target)
            out.append(np.stack([count_events(prob >= t, obs, valid) for t in THRESHOLDS]))
    return np.stack(out)


def log_mlflow(args: argparse.Namespace, results: dict) -> None:
    """Log parameters, headline metrics and artifacts of one run to a local MLflow store."""
    import mlflow

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("cold-cloud-nowcast")
    with mlflow.start_run(run_name=f"seed{args.seed}"):
        mlflow.log_params(
            {
                "steps": args.steps,
                "batch": args.batch,
                "crop": args.crop,
                "lr": args.lr,
                "seed": args.seed,
                "threshold": results["threshold"],
            }
        )
        for split in ("val", "test"):
            for model in ("cnn", "persistence"):
                for name, value in results[split][model]["scores"].items():
                    mlflow.log_metric(f"{split}_{model}_{name}", value)
        mlflow.log_artifact(str(args.out / "cnn_results.json"))
        mlflow.log_artifact(str(args.out / "cnn.pt"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mlflow", action="store_true", help="log the run to MLflow")
    parser.add_argument("--store", type=Path, default=Path("data/processed/goes.zarr"))
    parser.add_argument("--val-start", default="2026-09-07")
    parser.add_argument("--test-start", default="2026-09-09")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--crop", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("artifacts"))
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    ds = xr.open_zarr(args.store)
    times, bt = ds["time"].values, ds["bt"]
    splits = split_triplets(times, sample_triplets(times), args.val_start, args.test_start)
    n_train_frames = int(np.searchsorted(times, np.datetime64(args.val_start)))
    frames = bt.isel(time=slice(0, n_train_frames)).values
    print(f"train samples {len(splits['train'])}, frames in memory {frames.shape}")

    model = SmallUNet()
    optimiser = torch.optim.Adam(model.parameters(), lr=args.lr)
    start = time.perf_counter()
    model.train()
    for step in range(1, args.steps + 1):
        x, y, valid = make_batch(frames, splits["train"], rng, args.batch, args.crop)
        loss = F.binary_cross_entropy_with_logits(model(x), y, reduction="none")
        loss = (loss * valid).sum() / valid.sum()
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
        if step % 100 == 0:
            print(f"step {step}, loss {loss.item():.4f}, {time.perf_counter() - start:.0f} s")

    args.out.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.out / "cnn.pt")
    val_counts = predict_counts(model, bt, splits["val"])
    val_csi = [scores(val_counts[:, i].sum(axis=0))["csi"] for i in range(len(THRESHOLDS))]
    best = int(np.argmax(val_csi))
    print(f"threshold chosen on validation: {THRESHOLDS[best]}")

    results = {"threshold": float(THRESHOLDS[best]), "steps": args.steps, "seed": args.seed}
    for name in ("val", "test"):
        part = splits[name]
        cnn = (val_counts if name == "val" else predict_counts(model, bt, part))[:, best]
        persistence = evaluate_persistence(bt, part)
        blocks = block_ids(times, part)
        results[name] = {
            "samples": len(part),
            "cnn": {"scores": scores(cnn.sum(axis=0)), "ci95": block_bootstrap(cnn, blocks)},
            "persistence": {
                "scores": scores(persistence.sum(axis=0)),
                "ci95": block_bootstrap(persistence, blocks),
            },
            "cnn_minus_persistence_ci95": block_bootstrap_diff(cnn, persistence, blocks),
        }
        print(name, json.dumps(results[name], indent=2))
    (args.out / "cnn_results.json").write_text(json.dumps(results, indent=2))
    if args.mlflow:
        log_mlflow(args, results)


if __name__ == "__main__":
    main()
