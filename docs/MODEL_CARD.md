# Model card: small U-Net for 60-minute cold-cloud nowcasting

## Intended use

A research and demonstration model for predicting the cold-cloud mask (infrared window brightness temperature below 235 K) 60 minutes ahead, trained separately on GOES-19 ABI and on Meteosat SEVIRI. It is not an operational forecast or warning product.

## Model

- Architecture: three-level U-Net with batch normalisation, about 0.49 million parameters, one logit per pixel. The same architecture is used for both sensors.
- Input: four channels, the window channel and the water-vapour channel at *t*−15 and *t*, scaled as (BT − 250 K) / 30; missing pixels are set to 0. GOES: C13 (10.3 µm) and C08 (6.2 µm). SEVIRI: IR 10.8 µm and WV 6.2 µm.
- Output: probability that the window BT at *t*+60 is below 235 K; a threshold is applied afterwards.
- Training: binary cross-entropy on valid target pixels, Adam (learning rate 1e-3), batch of 16 random 128 × 128 crops, 2000 steps, three seeds (0, 1, 2). CPU only; about 14 minutes per GOES run and 13-20 minutes per SEVIRI run (Intel Core Ultra 7 165H).
- Threshold: chosen on the validation days to maximise CSI. GOES: 0.50, 0.20 and 0.25 for seeds 0, 1 and 2. SEVIRI: 0.20, 0.40 and 0.25. The CSI is flat over this range of thresholds, which is why different seeds choose different values.

## Evaluation

Test days 9–10 September 2026 (188 samples) for both sensors. Metrics are computed over all valid pixels: POD, FAR and CSI. Intervals come from a bootstrap that resamples 6-hour blocks (8 blocks in the test set). Two reference forecasts are scored the same way: persistence (the mask at *t*) and advection (the 10.8 or 10.3 µm field moved along DIS optical flow from the last two frames for 60 minutes, then thresholded at 235 K; the threshold is not tuned).

**Meteosat SEVIRI (West and Central Africa, 3 km pixels)**

| Run | POD | FAR | CSI (95% CI) | CSI minus advection (paired 95% CI) |
|---|---|---|---|---|
| Persistence | 0.695 | 0.309 | 0.531 (0.487-0.562) | -0.103 to -0.072 |
| Advection | 0.824 | 0.290 | 0.617 (0.577-0.643) | - |
| CNN seed 0 | 0.790 | 0.144 | 0.698 (0.665-0.721) | +0.073 to +0.093 |
| CNN seed 1 | 0.789 | 0.143 | 0.697 (0.656-0.723) | +0.069 to +0.091 |
| CNN seed 2 | 0.813 | 0.169 | 0.697 (0.658-0.722) | +0.070 to +0.090 |

**GOES-19 (western Atlantic, Caribbean and Florida, 2 km pixels)**

| Run | POD | FAR | CSI (95% CI) | CSI minus advection (paired 95% CI) |
|---|---|---|---|---|
| Persistence | 0.537 | 0.459 | 0.369 (0.328-0.401) | -0.109 to -0.085 |
| Advection | 0.664 | 0.390 | 0.466 (0.425-0.498) | - |
| CNN seed 0 | 0.658 | 0.284 | 0.522 (0.479-0.556) | +0.043 to +0.067 |
| CNN seed 1 | 0.689 | 0.326 | 0.517 (0.475-0.547) | +0.039 to +0.060 |
| CNN seed 2 | 0.674 | 0.294 | 0.526 (0.482-0.561) | +0.050 to +0.071 |

On both sensors the CNN beats persistence and advection in every seed. Advection already recovers about half (SEVIRI) to 60% (GOES) of the CNN's gain over persistence; the CNN adds about +0.08 CSI (SEVIRI) and +0.06 CSI (GOES) beyond motion, mainly through fewer false alarms. The two sensors are not a controlled comparison (different regions, land against ocean, pixel size and wavelength), so the higher SEVIRI scores should not be read as a sensor effect.

## Limitations

- Trained and tested on ten days of September 2026 in two regions, one mostly ocean and one mostly land. Performance in other regions and seasons is unknown, and each model is trained on a single sensor and region only.
- Validation and test days differ in cold-cloud fraction (GOES 8.1% and 4.2%; SEVIRI 4.4% and 3.4%), which shows that the score depends on the weather regime.
- The label is a brightness temperature threshold, not a convection or rainfall label.
- Pixel-wise scoring without spatial tolerance.
- The CNN threshold is tuned on validation days; the advection threshold is fixed at 235 K. A tuned or probabilistic advection baseline might narrow the gap.
- With only 8 test blocks from two days, the intervals are likely too narrow.
- No hyperparameter search, no early stopping.
- Probabilities are not calibrated, and no calibration analysis was done.
- Some SEVIRI files carry a quality flag "not OK"; these scenes were not analysed separately.

## Released weights

GitHub releases carry the seed-0 weights of the GOES and SEVIRI models (seed 0 by convention, the first run, not picked on test scores; the other seeds score within 0.01 CSI), a JSON file with the input convention and threshold, and checksums. `examples/predict.py` loads them and forecasts one frame.

## Reproduction

```
python -m eo_ml_ready.ingest --start 2026-09-01 --end 2026-09-11
python -m eo_ml_ready.train --steps 2000 --seed 0 --out artifacts/seed0 --mlflow
python -m eo_ml_ready.compare --store data/processed/goes.zarr --runs artifacts
```

For SEVIRI add `--source seviri --workers 2` to the ingest, and `--store data/processed/seviri.zarr` and `--out artifacts/seviri/seed0` to the training. Results of each run are written to `cnn_results.json`; with `--mlflow` they are also logged to a local MLflow store (`mlflow.db`).
