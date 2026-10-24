"""Build nowcasting samples (inputs at t-15 and t, target at t+60) from the Zarr store."""

from __future__ import annotations

import numpy as np
import xarray as xr

LAG = np.timedelta64(15, "m")
LEAD = np.timedelta64(60, "m")
THRESHOLD_K = 235.0
BLOCK_HOURS = 6


def sample_triplets(times: np.ndarray) -> np.ndarray:
    """Return (lag, now, lead) frame indices for every time with both neighbours present."""
    keep = np.flatnonzero(np.isin(times - LAG, times) & np.isin(times + LEAD, times))
    lag = np.searchsorted(times, times[keep] - LAG)
    lead = np.searchsorted(times, times[keep] + LEAD)
    return np.stack([lag, keep, lead], axis=1)


def split_triplets(
    times: np.ndarray, triplets: np.ndarray, val_start: str, test_start: str
) -> dict[str, np.ndarray]:
    """Split in time order; train and val targets must end before the next period starts."""
    origin = times[triplets[:, 1]]
    target = times[triplets[:, 2]]
    val_edge, test_edge = np.datetime64(val_start), np.datetime64(test_start)
    return {
        "train": triplets[(origin < val_edge) & (target < val_edge)],
        "val": triplets[(origin >= val_edge) & (origin < test_edge) & (target < test_edge)],
        "test": triplets[origin >= test_edge],
    }


def normalise(inputs_k: np.ndarray) -> np.ndarray:
    """Scale brightness temperatures to roughly [-2, 2] and set missing pixels to the mean."""
    return np.nan_to_num((inputs_k - 250.0) / 30.0, nan=0.0).astype(np.float32)


def block_ids(times: np.ndarray, triplets: np.ndarray) -> np.ndarray:
    """Return the 6-hour block number of each sample's origin time."""
    hours = times[triplets[:, 1]].astype("datetime64[h]").astype(np.int64)
    return hours // BLOCK_HOURS


def read_sample(bt: xr.DataArray, triplet: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return inputs (4, H, W) in K at t-15 and t, and the C13 target field at t+60 in K."""
    lag, now, lead = (int(i) for i in triplet)
    inputs = bt.isel(time=[lag, now]).values.astype(np.float32)
    target = bt.isel(time=lead, channel=0).values.astype(np.float32)
    return inputs.reshape(-1, *inputs.shape[-2:]), target


def cold_mask(field_k: np.ndarray) -> np.ndarray:
    """Return True where the brightness temperature is below the cold-cloud threshold."""
    with np.errstate(invalid="ignore"):
        return field_k < THRESHOLD_K
