"""Package a trained run for a GitHub release: weights, metadata JSON and a checksum file."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

SENSORS = {
    "goes19": {
        "channels": ["C13 (10.3 um)", "C08 (6.2 um)"],
        "region": "western Atlantic, Caribbean and Florida (about 14.6-36.7N, 84.6-60.5W)",
        "pixel_km": 2,
        "store": "data/processed/goes.zarr",
    },
    "seviri": {
        "channels": ["IR_108 (10.8 um)", "WV_062 (6.2 um)"],
        "region": "West and Central Africa (about 3.5S-26N, 14W-18E)",
        "pixel_km": 3,
        "store": "data/processed/seviri.zarr",
    },
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True, help="run folder with cnn.pt")
    parser.add_argument("--sensor", choices=sorted(SENSORS), required=True)
    parser.add_argument("--out", type=Path, default=Path("release"))
    args = parser.parse_args()

    results = json.loads((args.run / "cnn_results.json").read_text())
    seed = results["seed"]
    stem = f"eo-ml-ready-unet-{args.sensor}-seed{seed}"
    args.out.mkdir(parents=True, exist_ok=True)

    weights = args.out / f"{stem}.pt"
    weights.write_bytes((args.run / "cnn.pt").read_bytes())

    test = results["test"]
    info = SENSORS[args.sensor]
    metadata = {
        "name": stem,
        "model": "SmallUNet(in_channels=4, base=16), eo_ml_ready.model",
        "input": "channels [window(t-15), vapour(t-15), window(t), vapour(t)] in kelvin",
        "channel_names": info["channels"],
        "normalisation": "(BT - 250) / 30, missing pixels set to 0",
        "output": "logit of P(window BT at t+60 min < 235 K); apply a sigmoid, then the threshold",
        "threshold": results["threshold"],
        "threshold_note": "chosen on validation days to maximise CSI",
        "seed": seed,
        "steps": results["steps"],
        "trained_on": "1-6 September 2026, 15-minute steps",
        "region": info["region"],
        "pixel_km": info["pixel_km"],
        "test_days": "9-10 September 2026 (188 samples)",
        "test_scores": test["cnn"]["scores"],
        "test_persistence_scores": test["persistence"]["scores"],
        "torch_version": torch.__version__,
        "intended_use": "research and demonstration; not an operational forecast",
        "limitations": "ten days, one region; see docs/MODEL_CARD.md",
    }
    meta_path = args.out / f"{stem}.json"
    meta_path.write_text(json.dumps(metadata, indent=2) + "\n")

    lines = [f"{sha256(path)}  {path.name}" for path in (weights, meta_path)]
    with (args.out / "SHA256SUMS.txt").open("a") as handle:
        handle.write("\n".join(lines) + "\n")
    print(f"Wrote {weights.name} ({weights.stat().st_size / 1e6:.1f} MB) and {meta_path.name}")


if __name__ == "__main__":
    main()
