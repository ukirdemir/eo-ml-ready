# eo-ml-ready

From geostationary satellite imagery to an ML-ready dataset, and a small nowcasting model on top of it.

The repo does two things:

1. **Pipeline.** Fetch scenes through a provider-neutral interface, crop and stack them into a time-regular Zarr store with geolocation metadata, and build leakage-safe train/validation/test samples.
2. **Detector.** Predict where deep cold cloud (a proxy for deep convection) will be 60 minutes ahead, and compare a small CNN against persistence with categorical scores and block-bootstrap intervals.

## Task

Input: brightness temperature at 10.3 µm (C13) and 6.2 µm (C08) at times *t* and *t*−15 min.
Target: the cold-cloud mask (C13 < 235 K) at *t*+60 min, per pixel.

## Results

Test set: 9–10 September 2026 (188 samples). The threshold of the CNN is tuned on validation days only. Three seeds, 2000 steps each.

| Run | Threshold | POD | FAR | CSI |
|---|---|---|---|---|
| CNN seed 0 | 0.50 | 0.658 | 0.284 | 0.522 |
| CNN seed 1 | 0.20 | 0.689 | 0.326 | 0.517 |
| CNN seed 2 | 0.25 | 0.674 | 0.294 | 0.526 |
| CNN mean (min-max CSI) | | | | 0.522 (0.517-0.526) |
| Persistence | | 0.537 | 0.459 | 0.369 (95% CI 0.328-0.401) |

POD = probability of detection, FAR = false-alarm ratio, CSI = critical success index. The CNN improves CSI by about 0.15 over persistence in every seed. Read the numbers together with the limitations below: the test period is short.

## Quick start

```
pip install -e ".[dev,train,track]"
pytest

python -m eo_ml_ready.ingest --start 2026-09-01 --end 2026-09-11
python -m eo_ml_ready.baseline
python -m eo_ml_ready.train --steps 2000 --seed 0 --out artifacts/seed0 --mlflow
python -m eo_ml_ready.summarise --out docs/RESULTS.md
```

Ingest downloads about 7 GB in total, crops each scene and deletes the raw files. The store is a few GB on disk. Training takes about 14 minutes on a CPU and needs about 2.4 GB of RAM.

A Docker image (`docker build -t eo-ml-ready .`) runs the same modules, for example `docker run -v ${PWD}/data:/app/data eo-ml-ready eo_ml_ready.ingest --start ...`.

## Layout

| Path | Purpose |
|---|---|
| `src/eo_ml_ready/fetch/` | Provider interface, GOES-on-AWS adapter, EUMETSAT adapter |
| `src/eo_ml_ready/ingest.py` | Crop, stack and write the Zarr store; resumes by day |
| `src/eo_ml_ready/geo.py` | Scan angles to latitude/longitude |
| `src/eo_ml_ready/samples.py` | Triplets, time-ordered splits, 6-hour block ids |
| `src/eo_ml_ready/metrics.py` | POD/FAR/CSI and (paired) block bootstrap |
| `src/eo_ml_ready/baseline.py` | Persistence forecast |
| `src/eo_ml_ready/model.py`, `train.py` | Small U-Net, training, threshold tuning, evaluation |
| `scripts/` | Inspect and plot single scenes |
| `fallback/` | Diagnostics written for the EUMETSAT Data Store (scene inspection, download probe, adapter test) |
| `docs/` | Dataset card, model card, tutorial, results table, figures |

## Design choices

- **Split in time, not at random.** Consecutive frames are highly correlated. Training samples whose target time falls in the validation period are dropped, and the same for validation and test.
- **Threshold tuned on validation only.** The test days are touched once per run.
- **Uncertainty by time blocks.** Pixels and neighbouring time steps are not independent, so the bootstrap resamples 6-hour blocks. With two test days there are only 8 blocks, so the intervals are likely too narrow (see limitations).
- **Persistence as the baseline.** At 60 minutes ahead it is a strong reference for slowly evolving cloud shields.
- **Provider-neutral fetching.** Each source implements `list_scenes` and `download`; the rest of the pipeline sees only `SceneRef` objects.

## Limitations

- **Short period, one region.** Ten days of September 2026, over the western Atlantic, the Caribbean and Florida (about 14.6–36.7°N, 84.6–60.5°W). The area is mostly ocean. Conclusions about other regions, seasons or land convection are not supported.
- **Two test days.** The cold-cloud fraction differs between validation (8.1%) and test (4.2%). The block bootstrap on 8 blocks understates the true uncertainty; treat the confidence intervals as indicative.
- **The target is a proxy.** BT < 235 K at 10.3 µm marks cold cloud tops, which includes thick cirrus and frontal cloud as well as convection. It is not a convection or precipitation label.
- **Pixel-wise scores.** There is no spatial tolerance and no object-based verification, which penalises small displacements of correctly forecast cells.
- **Training budget.** One architecture, fixed 2000 steps, no hyperparameter search, no early stopping.
- **Per-day and land/sea breakdowns are not analysed.**
- **Not an operational system.**

## Data

Development data are GOES-19 ABI Level 2 cloud and moisture imagery (ABI-L2-CMIPC) from the open NOAA bucket `noaa-goes19` on AWS, CONUS sector, 5-minute scans subsampled to 15 minutes. Details are in `docs/DATASET_CARD.md`.

An adapter for EUMETSAT MSG SEVIRI (`eumdac`, collection `EO:EUM:DAT:MSG:HRSEVIRI`) is included in `fetch/eumetsat.py`. Listing and downloading a full-disk scene (187 MB zip) and reading it with satpy (`seviri_l1b_native`: 12 channels, 3712 x 3712 pixels, IR 10.8 µm and WV 6.2 µm in kelvin) were tested on 15 September 2026 with `fallback/test_seviri_adapter.py` and `fallback/inspect_scene.py`. The ingest step does not yet read SEVIRI, so no Zarr store has been built from it. The SEVIRI IR 10.8 µm channel is the natural counterpart of ABI C13, and 6.2 µm of C08.

## Roadmap

- Add a SEVIRI reader to the ingest step, build a SEVIRI Zarr store and rerun the nowcast on it.
- Longer period and several seasons; per-day and land/sea evaluation.
- Object-based verification and spatial tolerance.
- Precipitation targets from radar.
