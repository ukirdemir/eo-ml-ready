"""Advection baseline: a moving cold cloud should be forecast better than by persistence."""

import numpy as np
import pytest

pytest.importorskip("cv2")

from eo_ml_ready.baseline import advect, advection_mask, persistence_mask
from eo_ml_ready.metrics import count_events, scores
from eo_ml_ready.samples import cold_mask


def cloud(shift: int, size: int = 160) -> np.ndarray:
    """A textured cold cloud (about 215 K) on a 295 K background, moved east by ``shift``."""
    yy, xx = np.mgrid[:size, :size]
    moving_x = xx - shift  # the texture moves with the cloud
    texture = 8.0 * np.sin(0.35 * moving_x) * np.sin(0.3 * yy)
    texture += 5.0 * np.cos(0.21 * moving_x + yy / 9)
    inside = (yy - 80) ** 2 + (xx - (50 + shift)) ** 2 < 22**2
    return np.where(inside, 215.0 + texture, 295.0 + 0.3 * texture)


def csi(prediction: np.ndarray, truth: np.ndarray) -> float:
    return scores(count_events(prediction, truth, np.ones_like(truth, dtype=bool)))["csi"]


def test_advect_shifts_field_by_flow():
    field = cloud(0)
    flow = np.zeros((*field.shape, 2), dtype=np.float32)
    flow[..., 0] = 2.0  # two pixels east per step
    moved = advect(field, flow, steps=4)
    assert np.mean(moved < 235.0) == pytest.approx(np.mean(cloud(8) < 235.0), abs=0.01)


def test_advection_beats_persistence_for_moving_cloud():
    # frames 15 minutes apart, cloud moving 3 pixels per step; the target is 4 steps later
    inputs = np.stack([cloud(0), np.full((160, 160), 250.0), cloud(3), np.full((160, 160), 250.0)])
    truth = cold_mask(cloud(3 + 12))
    assert csi(advection_mask(inputs), truth) > csi(persistence_mask(inputs), truth) + 0.1
