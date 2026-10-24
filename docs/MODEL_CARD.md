# Model card: small U-Net for 60-minute cold-cloud nowcasting

## Intended use

A research and demonstration baseline for predicting the cold-cloud mask (10.3 µm brightness temperature below 235 K) 60 minutes ahead. It is not an operational forecast or warning product.

## Model

- Architecture: three-level U-Net with batch normalisation, about 0.49 million parameters, one logit per pixel.
- Input: four channels, C13 and C08 at *t*−15 and *t*, scaled as (BT − 250 K) / 30; missing pixels are set to 0.
- Output: probability that BT at *t*+60 is below 235 K; a threshold is applied afterwards.
- Training: binary cross-entropy on valid target pixels, Adam (learning rate 1e-3), batch of 16 random 128 × 128 crops, 2000 steps, three seeds (0, 1, 2). CPU, about 14 minutes per run.
- Threshold: chosen on the validation days to maximise CSI. The chosen thresholds were 0.50, 0.20 and 0.25 for the three seeds.

## Evaluation

Test days 9–10 September 2026 (188 samples). Metrics are computed over all valid pixels: POD, FAR and CSI. Intervals come from a bootstrap that resamples 6-hour blocks (8 blocks in the test set).

| Run | POD | FAR | CSI |
|---|---|---|---|
| CNN seed 0 | 0.658 | 0.284 | 0.522 |
| CNN seed 1 | 0.689 | 0.326 | 0.517 |
| CNN seed 2 | 0.674 | 0.294 | 0.526 |
| Persistence | 0.537 | 0.459 | 0.369 (95% CI 0.328-0.401) |

The paired bootstrap interval for the CSI difference (CNN minus persistence) was roughly 0.14–0.17 for every seed. With 8 blocks from two days, this interval is likely too narrow.

## Limitations

- Trained and tested on ten days over one region, mostly ocean. Performance elsewhere is unknown.
- Validation and test days differ in cold-cloud fraction (8.1% vs 4.2%), which shows that the score depends on the weather regime.
- The label is a brightness temperature threshold, not a convection or rainfall label.
- Pixel-wise scoring without spatial tolerance.
- No hyperparameter search, no early stopping, no comparison with optical-flow or other advection baselines. An advection baseline would be a stronger reference than persistence and is the next comparison to add.
- Probabilities are not calibrated, and no calibration analysis was done.

## Reproduction

```
python -m eo_ml_ready.ingest --start 2026-09-01 --end 2026-09-11
python -m eo_ml_ready.train --steps 2000 --seed 0 --out artifacts/seed0 --mlflow
```

Results of each run are written to `cnn_results.json`; with `--mlflow` they are also logged to a local MLflow store (`mlflow.db`).
