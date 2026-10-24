"""Shape and batch tests for the CNN (skipped when torch is not installed)."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from eo_ml_ready.model import SmallUNet  # noqa: E402
from eo_ml_ready.train import make_batch  # noqa: E402


def test_unet_keeps_spatial_size():
    model = SmallUNet().eval()
    out = model(torch.zeros(2, 4, 64, 64))
    assert out.shape == (2, 1, 64, 64)


def test_make_batch_shapes_and_masks():
    rng = np.random.default_rng(0)
    frames = np.full((6, 2, 32, 32), 250.0, dtype=np.float16)
    frames[5, 0, :8] = 200.0
    frames[5, 0, 8, 8] = np.nan
    triplets = np.array([[0, 1, 5]])
    x, y, valid = make_batch(frames, triplets, rng, batch=3, crop=16)
    assert x.shape == (3, 4, 16, 16)
    assert y.shape == valid.shape == (3, 1, 16, 16)
    assert set(torch.unique(y).tolist()) <= {0.0, 1.0}
