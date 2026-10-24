"""Summarise CNN results across seeds as a markdown table."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def table(runs: list[dict]) -> str:
    """Return a markdown table of test scores per seed, with persistence and the seed range."""
    rows = [
        "| Run | Threshold | POD | FAR | CSI |",
        "|---|---|---|---|---|",
    ]
    for run in runs:
        s = run["test"]["cnn"]["scores"]
        rows.append(
            f"| CNN seed {run['seed']} | {run['threshold']:.2f} | "
            f"{s['pod']:.3f} | {s['far']:.3f} | {s['csi']:.3f} |"
        )
    csi = [run["test"]["cnn"]["scores"]["csi"] for run in runs]
    rows.append(
        f"| CNN mean (min-max CSI) | | | | {np.mean(csi):.3f} ({min(csi):.3f}-{max(csi):.3f}) |"
    )
    p = runs[0]["test"]["persistence"]["scores"]
    ci = runs[0]["test"]["persistence"]["ci95"]["csi"]
    rows.append(
        f"| Persistence | | {p['pod']:.3f} | {p['far']:.3f} | "
        f"{p['csi']:.3f} (95% CI {ci[0]:.3f}-{ci[1]:.3f}) |"
    )
    return "\n".join(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    runs = [json.loads(p.read_text()) for p in sorted(args.dir.glob("seed*/cnn_results.json"))]
    if not runs:
        raise SystemExit(f"No seed*/cnn_results.json under {args.dir}")
    text = table(runs)
    print(text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n")


if __name__ == "__main__":
    main()
