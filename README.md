# eo-ml-ready

From geostationary satellite imagery to an ML-ready dataset, and a small nowcasting model on top of it.

The repo does two things:

1. **Pipeline.** Fetch scenes through a provider-neutral interface, crop and stack them into a time-regular Zarr store with geolocation metadata, and build leakage-safe train/validation/test samples.
2. **Detector.** Predict where deep cold cloud (a proxy for deep convection) will be 60 minutes ahead, and compare a small CNN against persistence and optical-flow advection with categorical scores and block-bootstrap intervals.

The same pipeline and the same evaluation have been run on two sensors: GOES-19 ABI over the western Atlantic and Caribbean, and Meteosat SEVIRI (EUMETSAT Data Store) over West and Central Africa.

## Task

Input: brightness temperature in an infrared window channel (GOES C13 10.3 µm; SEVIRI IR 10.8 µm) and a water-vapour channel (C08 or WV 6.2 µm) at times *t* and *t*−15 min.
Target: the cold-cloud mask (window channel < 235 K) at *t*+60 min, per pixel.

## Results

Test set: 9–10 September 2026 (188 samples). The threshold of the CNN is tuned on validation days only. Three seeds, 2000 steps each. Intervals are 95% block-bootstrap intervals; the last column is the paired interval of the score difference to advection (same resampled blocks).

**Meteosat SEVIRI, West and Central Africa (about 3.5°S–26°N, 14°W–18°E, 3 km pixels)**

| Method | POD | FAR | CSI (95% CI) | CSI minus advection |
|---|---|---|---|---|
| Persistence | 0.695 | 0.309 | 0.531 (0.487-0.562) | -0.103 to -0.072 |
| Advection (DIS optical flow) | 0.824 | 0.290 | 0.617 (0.577-0.643) | - |
| CNN seed 0 | 0.790 | 0.144 | 0.698 (0.665-0.721) | +0.073 to +0.093 |
| CNN seed 1 | 0.789 | 0.143 | 0.697 (0.656-0.723) | +0.069 to +0.091 |
| CNN seed 2 | 0.813 | 0.169 | 0.697 (0.658-0.722) | +0.070 to +0.090 |

**GOES-19, western Atlantic, Caribbean and Florida (about 14.6–36.7°N, 84.6–60.5°W, 2 km pixels)**

| Method | POD | FAR | CSI (95% CI) | CSI minus advection |
|---|---|---|---|---|
| Persistence | 0.537 | 0.459 | 0.369 (0.328-0.401) | -0.109 to -0.085 |
| Advection (DIS optical flow) | 0.664 | 0.390 | 0.466 (0.425-0.498) | - |
| CNN seed 0 | 0.658 | 0.284 | 0.522 (0.479-0.556) | +0.043 to +0.067 |
| CNN seed 1 | 0.689 | 0.326 | 0.517 (0.475-0.547) | +0.039 to +0.060 |
| CNN seed 2 | 0.674 | 0.294 | 0.526 (0.482-0.561) | +0.050 to +0.071 |

POD = probability of detection, FAR = false-alarm ratio, CSI = critical success index.

What the results say:

- On both sensors the CNN beats persistence and also optical-flow advection, in every seed, with paired intervals above zero. The seeds agree closely (SEVIRI CSI 0.697-0.698, GOES 0.517-0.526).
- Advection explains a large part of the gain over persistence (about half on SEVIRI, about 60% on GOES). Persistence alone overstates what the network learns; the added value of the CNN over motion is about +0.08 CSI on SEVIRI and +0.06 on GOES.
- The CNN gains mainly by fewer false alarms (SEVIRI FAR 0.14-0.17 against 0.29 for advection), not by detecting more cold cloud.
- Scores are higher on SEVIRI for every method, but the two sensors are not a controlled comparison: region (land against ocean), pixel size (3 km against 2 km), wavelength (10.8 against 10.3 µm) and the cloud systems differ (large, slow organised systems over Africa against small cells over the ocean). Do not read the difference as a sensor effect.
- The CNN threshold is tuned on validation days while advection uses the fixed 235 K threshold, which favours the CNN somewhat. Advection was not tuned.

Read the numbers together with the limitations below: the test period is two days.

**Released weights.** The seed-0 models for GOES-19 and SEVIRI (about 2 MB each) are attached to the GitHub release, with a JSON file for each (input convention, threshold, test scores) and checksums. To run one on a frame of your own store:

```
python examples/predict.py --weights eo-ml-ready-unet-seviri-seed0.pt --store data/processed/seviri.zarr --origin 2026-09-09T12:00
```

The matching `.json` file must sit next to the weights. Seed 0 is the first run, taken by convention and not picked for its test score; the other seeds score within 0.01 CSI of it. The models were trained on ten days of one region each and are research baselines, not forecast products (see `docs/MODEL_CARD.md`).

## Scaling up

The pipeline was built and tested on ten days and two 1024 x 1024 windows, on one CPU laptop. Going from there to months or years of data, full-disk fields or a new sensor needs a few concrete changes. They are not implemented here.

- **Data volume.** One SEVIRI day is 96 scenes of 187 MB (about 18 GB downloaded, 30 minutes on a home connection); a year would be about 6.5 TB and about 8 days of continuous downloading. The ingest already resumes by day and deletes raw files after cropping, so it can run in parallel over day ranges on separate machines or workers. A server-side subset (the EUMETSAT Data Tailor, or GOES regional files) would cut the transfer.
- **Store layout.** The Zarr store (float16, chunks of one time step and 256 x 256 pixels) holds a few GB for ten days and scales linearly in time and area. For a year it would be about 150 GB; appending by day and chunking by time already match this. Several years or full-disk grids would need sharded or object-store storage and a catalogue of the time ranges present.
- **Training.** Training now loads the training frames into memory (about 2.4 GB for six days). Longer periods need lazy, chunk-aligned reading with a data loader and shuffling across days, and a GPU. The model (0.49 M parameters) is not the limit; the input pipeline is.
- **Evaluation.** Several seasons and regions allow per-regime scores and a bootstrap with many more blocks than the 8 used here, so the intervals become meaningful.
- **New sensors.** A new source needs a provider (list and download) and a reader that returns the channels in a fixed order (`Source` in `ingest.py`). Meteosat Third Generation (FCI) would follow the SEVIRI path.

## Quick start

GOES (open data, no account):

```
pip install -e ".[dev,train,track,advection]"
pytest

python -m eo_ml_ready.ingest --start 2026-09-01 --end 2026-09-11
python -m eo_ml_ready.baseline --method persistence
python -m eo_ml_ready.baseline --method advection
python -m eo_ml_ready.train --steps 2000 --seed 0 --out artifacts/seed0 --mlflow
python -m eo_ml_ready.summarise --out docs/RESULTS.md
python -m eo_ml_ready.compare --store data/processed/goes.zarr --runs artifacts
```

SEVIRI (needs an EUMETSAT Earth Observation Portal account, `eumdac --set-credentials`, and the Meteosat Core data licence):

```
pip install -e ".[dev,train,eumetsat,advection]"

python -m eo_ml_ready.ingest --source seviri --start 2026-09-01 --end 2026-09-11 --workers 2
python scripts/check_store.py
python -m eo_ml_ready.baseline --method persistence --store data/processed/seviri.zarr --out artifacts/seviri/persistence.json
python -m eo_ml_ready.baseline --method advection --store data/processed/seviri.zarr --out artifacts/seviri/advection.json
python -m eo_ml_ready.train --store data/processed/seviri.zarr --seed 0 --out artifacts/seviri/seed0
python -m eo_ml_ready.compare --store data/processed/seviri.zarr --runs artifacts/seviri
```

GOES ingest downloads about 7 GB in total. SEVIRI ingest downloads about 18 GB per day (187 MB per scene, 96 scenes) and took about 30 minutes per day on a home connection; raw files are deleted after cropping. Each store is a few GB on disk. Training takes 13-20 minutes per seed on a CPU (Intel Core Ultra 7 165H) and needs about 2.4 GB of RAM.

A Docker image (`docker build -t eo-ml-ready .`) runs the same modules, for example `docker run -v ${PWD}/data:/app/data eo-ml-ready eo_ml_ready.ingest --start ...`.

## Layout

| Path | Purpose |
|---|---|
| `src/eo_ml_ready/fetch/` | Provider interface, GOES-on-AWS adapter, EUMETSAT adapter |
| `src/eo_ml_ready/ingest.py` | Crop, stack and write the Zarr store for a source (GOES or SEVIRI); resumes by day |
| `src/eo_ml_ready/seviri.py` | SEVIRI native reader (satpy): north-up flip, crop, corner coordinates |
| `src/eo_ml_ready/geo.py` | GOES scan angles to latitude/longitude |
| `src/eo_ml_ready/samples.py` | Triplets, time-ordered splits, 6-hour block ids |
| `src/eo_ml_ready/metrics.py` | POD/FAR/CSI and (paired) block bootstrap |
| `src/eo_ml_ready/baseline.py` | Persistence and optical-flow advection forecasts |
| `src/eo_ml_ready/model.py`, `train.py` | Small U-Net, training, threshold tuning, evaluation |
| `src/eo_ml_ready/compare.py` | CNN seeds against persistence and advection with paired intervals |
| `scripts/` | Inspect and plot single scenes; check a store (`check_store.py`, `plot_store_frame.py`); package weights for a release (`export_weights.py`) |
| `examples/predict.py` | Load released weights and forecast one frame |
| `fallback/` | Diagnostics written for the EUMETSAT Data Store (scene inspection, download probe, adapter test) |
| `docs/` | Dataset card, model card, tutorial, results table, figures |

## Design choices

- **Split in time, not at random.** Consecutive frames are highly correlated. Training samples whose target time falls in the validation period are dropped, and the same for validation and test.
- **Threshold tuned on validation only.** The test days are touched once per run.
- **Uncertainty by time blocks.** Pixels and neighbouring time steps are not independent, so the bootstrap resamples 6-hour blocks. With two test days there are only 8 blocks, so the intervals are likely too narrow (see limitations).
- **Two baselines.** Persistence is the minimum reference; optical-flow advection (DIS flow from the last two frames, applied for four more steps) separates what the network learns beyond cloud motion. Advection recovers about half of the CNN's gain over persistence.
- **Provider-neutral fetching.** Each source implements `list_scenes` and `download`; the rest of the pipeline sees only `SceneRef` objects.

## Limitations

- **Short period, two regions.** Ten days of September 2026. GOES: western Atlantic, Caribbean and Florida (mostly ocean). SEVIRI: West and Central Africa (land and Gulf of Guinea). Conclusions about other regions or seasons are not supported, and the two sensors are not a controlled comparison.
- **Two test days.** The cold-cloud fraction differs between validation and test (GOES 8.1% and 4.2%; SEVIRI 4.4% and 3.4%). The block bootstrap on 8 blocks understates the true uncertainty; treat the confidence intervals as indicative.
- **Advection is not tuned.** It uses the fixed 235 K threshold, while the CNN threshold is tuned on validation days. A tuned or probabilistic advection baseline might narrow the gap.
- **SEVIRI quality flags.** Some SEVIRI files carry a quality flag "not OK" (satpy warning). The store has no missing pixels and no frame is dropped, but the flagged scenes were not analysed separately.
- **The target is a proxy.** BT < 235 K at 10.3 µm marks cold cloud tops, which includes thick cirrus and frontal cloud as well as convection. It is not a convection or precipitation label.
- **Pixel-wise scores.** There is no spatial tolerance and no object-based verification, which penalises small displacements of correctly forecast cells.
- **Training budget.** One architecture, fixed 2000 steps, no hyperparameter search, no early stopping.
- **Per-day and land/sea breakdowns are not analysed.**
- **Not an operational system.**

## Data

Development data are GOES-19 ABI Level 2 cloud and moisture imagery (ABI-L2-CMIPC) from the open NOAA bucket `noaa-goes19` on AWS, CONUS sector, 5-minute scans subsampled to 15 minutes. Details are in `docs/DATASET_CARD.md`.

The same ten days of Meteosat SEVIRI Level 1.5 (collection `EO:EUM:DAT:MSG:HRSEVIRI`) come from the EUMETSAT Data Store through `fetch/eumetsat.py` (`eumdac`). Full-disk native files (187 MB zip, 3712 x 3712 pixels, 15-minute steps) are read with satpy (`seviri_l1b_native`) by `seviri.py`, flipped to north-up, and cropped to a 1024 x 1024 window (rows 960-1984, columns 1400-2424 of the full disk, with the corner coordinates stored in the store attributes). IR 10.8 µm is the counterpart of ABI C13, and WV 6.2 µm of C08. Values outside 100-350 K are set to missing. The data are subject to the EUMETSAT Data Policy; see `docs/DATASET_CARD.md`. `fallback/` holds the scripts used to diagnose the Data Store licence and to test the adapter.

## Roadmap

- Longer period and several seasons; per-day and land/sea evaluation.
- Tuned or probabilistic advection baseline; blending of advection and CNN.
- Meteosat Third Generation (FCI) data through the same adapter.
- Object-based verification and spatial tolerance.
- Precipitation targets from radar.
