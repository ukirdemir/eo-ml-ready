"""Categorical scores (POD, FAR, CSI) and a block bootstrap for their uncertainty."""

from __future__ import annotations

import numpy as np


def count_events(pred: np.ndarray, obs: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Return [hits, misses, false alarms, correct negatives] over valid pixels."""
    pred, obs = pred[valid], obs[valid]
    return np.array(
        [
            np.sum(pred & obs),
            np.sum(~pred & obs),
            np.sum(pred & ~obs),
            np.sum(~pred & ~obs),
        ],
        dtype=np.int64,
    )


def scores(counts: np.ndarray) -> dict[str, float]:
    """Return POD, FAR and CSI from [hits, misses, false alarms, correct negatives]."""
    hits, misses, false_alarms = (float(c) for c in counts[:3])
    with np.errstate(invalid="ignore", divide="ignore"):
        return {
            "pod": hits / (hits + misses) if hits + misses else float("nan"),
            "far": false_alarms / (hits + false_alarms) if hits + false_alarms else float("nan"),
            "csi": hits / (hits + misses + false_alarms)
            if hits + misses + false_alarms
            else float("nan"),
        }


def block_bootstrap_diff(
    counts_a: np.ndarray,
    counts_b: np.ndarray,
    blocks: np.ndarray,
    n_boot: int = 2000,
    seed: int = 0,
) -> dict[str, tuple[float, float]]:
    """Return 95% intervals for score(a) - score(b) with the same resampled blocks."""
    unique = np.unique(blocks)
    sums_a = np.stack([counts_a[blocks == b].sum(axis=0) for b in unique])
    sums_b = np.stack([counts_b[blocks == b].sum(axis=0) for b in unique])
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(unique), size=(n_boot, len(unique)))
    diffs = []
    for row in draws:
        a, b = scores(sums_a[row].sum(axis=0)), scores(sums_b[row].sum(axis=0))
        diffs.append({name: a[name] - b[name] for name in a})
    return {
        name: (
            float(np.nanpercentile([d[name] for d in diffs], 2.5)),
            float(np.nanpercentile([d[name] for d in diffs], 97.5)),
        )
        for name in ("pod", "far", "csi")
    }


def block_bootstrap(
    counts: np.ndarray, blocks: np.ndarray, n_boot: int = 2000, seed: int = 0
) -> dict[str, tuple[float, float]]:
    """Return 95% intervals for POD, FAR and CSI by resampling whole time blocks."""
    unique = np.unique(blocks)
    block_sums = np.stack([counts[blocks == b].sum(axis=0) for b in unique])
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(unique), size=(n_boot, len(unique)))
    samples = [scores(block_sums[row].sum(axis=0)) for row in draws]
    return {
        name: (
            float(np.nanpercentile([s[name] for s in samples], 2.5)),
            float(np.nanpercentile([s[name] for s in samples], 97.5)),
        )
        for name in ("pod", "far", "csi")
    }
