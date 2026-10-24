# Tutorial: from raw scenes to a scored nowcast

This walks through the pipeline with the default settings. Everything runs on a CPU.

## 1. Install

```
pip install -e ".[dev,train,track]"
pytest
```

## 2. Look at one scene

```
python scripts/inspect_goes.py --time 2026-09-15T18:00
python scripts/plot_scene.py data/raw/goes/<a C13 file>.nc
```

The first script downloads one scan per channel and prints shape, units and value ranges. The second draws the 10.3 µm field with the 235 K contour, which is the cold-cloud label used later.

## 3. Build the dataset

```
python -m eo_ml_ready.ingest --start 2026-09-01 --end 2026-09-11
```

For each 15-minute time the script picks the nearest scan within 3 minutes for both channels, downloads them in parallel, crops the window, writes one day to Zarr and deletes the raw files. Running the command again skips days that are already stored. Check the result with:

```
python -c "import xarray as xr; ds = xr.open_zarr('data/processed/goes.zarr'); print(ds); print(ds.attrs)"
```

## 4. Understand the samples

A sample is a triplet of frame indices (*t*−15, *t*, *t*+60). Splits are chronological by origin day. Samples whose target falls into the next period are dropped. `samples.py` also assigns each sample to a 6-hour block, which the bootstrap uses.

## 5. Score the baseline

```
python -m eo_ml_ready.baseline
```

Persistence uses the cold-cloud mask at *t* as the forecast for *t*+60 and reports POD, FAR and CSI with block-bootstrap intervals.

## 6. Train and compare

```
python -m eo_ml_ready.train --steps 2000 --seed 0 --out artifacts/seed0 --mlflow
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

The run prints validation and test scores for the CNN and for persistence, and the paired interval for their difference. To check seed sensitivity, repeat with `--seed 1` and `--seed 2` and different `--out` folders, then:

```
python -m eo_ml_ready.summarise --out docs/RESULTS.md
```

## 7. Add another data source

Implement the `Provider` interface from `eo_ml_ready/fetch/base.py`: `list_scenes(start, end, channel)` returning `SceneRef` objects sorted by time, and `download(ref, dest)` returning a local path. The ingest step also needs a reader that returns brightness temperature and scan-angle coordinates for the window. `fetch/eumetsat.py` is a starting point for MSG SEVIRI.
