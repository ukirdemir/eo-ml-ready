"""Tests for sample indexing, splits, categorical scores and the block bootstrap."""

import numpy as np
import pytest

from eo_ml_ready.metrics import block_bootstrap, count_events, scores
from eo_ml_ready.samples import block_ids, cold_mask, sample_triplets, split_triplets


def regular_times(n, start="2026-09-01T00:00"):
    return np.datetime64(start, "ns") + np.arange(n) * np.timedelta64(15, "m")


def test_triplets_need_lag_and_lead():
    times = regular_times(10)
    triplets = sample_triplets(times)
    assert triplets[0].tolist() == [0, 1, 5]
    assert triplets[-1].tolist() == [4, 5, 9]
    assert len(triplets) == 5


def test_triplets_skip_gaps():
    times = np.delete(regular_times(10), 3)
    triplets = sample_triplets(times)
    assert all(times[lead] - times[now] == np.timedelta64(60, "m") for _, now, lead in triplets)


def test_split_is_chronological_and_disjoint():
    times = regular_times(96 * 3)
    triplets = sample_triplets(times)
    splits = split_triplets(times, triplets, "2026-09-02", "2026-09-03")
    assert sum(len(s) for s in splits.values()) < len(triplets)
    assert times[splits["train"][:, 2]].max() < np.datetime64("2026-09-02")
    assert times[splits["val"][:, 2]].max() < np.datetime64("2026-09-03")
    assert times[splits["test"][:, 1]].min() >= np.datetime64("2026-09-03")


def test_block_ids_group_six_hours():
    times = regular_times(96)
    ids = block_ids(times, np.array([[0, 0, 0], [0, 23, 0], [0, 24, 0]]))
    assert ids[0] == ids[1]
    assert ids[2] == ids[0] + 1


def test_counts_and_scores():
    pred = np.array([True, True, False, False])
    obs = np.array([True, False, True, False])
    counts = count_events(pred, obs, np.ones(4, dtype=bool))
    assert counts.tolist() == [1, 1, 1, 1]
    result = scores(counts)
    assert result["pod"] == pytest.approx(0.5)
    assert result["far"] == pytest.approx(0.5)
    assert result["csi"] == pytest.approx(1 / 3)


def test_invalid_pixels_are_ignored():
    pred = np.array([True, True])
    obs = np.array([True, False])
    counts = count_events(pred, obs, np.array([True, False]))
    assert counts.tolist() == [1, 0, 0, 0]


def test_cold_mask_treats_nan_as_not_cold():
    assert cold_mask(np.array([200.0, 250.0, np.nan])).tolist() == [True, False, False]


def test_bootstrap_interval_contains_point_estimate():
    rng = np.random.default_rng(1)
    counts = rng.integers(5, 50, size=(40, 4))
    blocks = np.repeat(np.arange(10), 4)
    point = scores(counts.sum(axis=0))["csi"]
    low, high = block_bootstrap(counts, blocks, n_boot=300)["csi"]
    assert low <= point <= high
