# Dataset card: cropped GOES-19 ABI brightness temperature, 1–10 September 2026

## Summary

A time-regular Zarr store of brightness temperature at two infrared channels over a fixed 1024 × 1024 pixel window, built by `python -m eo_ml_ready.ingest`.

## Source

- Satellite and product: GOES-19 ABI Level 2 Cloud and Moisture Imagery, CONUS sector (`ABI-L2-CMIPC`), from the open bucket `noaa-goes19` on AWS.
- Channels: C13 (10.3 µm, clean IR window) and C08 (6.2 µm, upper-level water vapour). The product already provides brightness temperature in kelvin.
- Native cadence 5 minutes; the store keeps the scan nearest to each 15-minute time within ±3 minutes.
- Nominal pixel size about 2 km at the sub-satellite point.

## Contents

| Item | Value |
|---|---|
| Variable | `bt` (time, channel, y, x), float16, kelvin, NaN where missing |
| Period | 2026-09-01 00:00 to 2026-09-10 23:45 UTC |
| Time steps | 960 (a step is kept only if both channels are present) |
| Window | rows 476–1500, columns 1400–2424 of the source grid |
| Corners (lat, lon) | NW 36.6, −84.6; NE 36.7, −60.5; SW 14.6, −82.7; SE 14.6, −63.4 |
| Coordinates | `time` (UTC), `channel`, `y`, `x` (scan angle in radians) |
| Attributes | title, source, channels, units, window pixels, corner latitudes and longitudes |
| Chunks | (1, 2, 256, 256) |

Float16 quantises brightness temperature in steps of roughly 0.06–0.12 K at 200–300 K. That is fine for thresholding at 235 K, but the store is not suitable for radiometric work that needs the original precision.

## Samples and splits

A sample needs frames at *t*−15 min, *t* and *t*+60 min. Splits are by origin time:

| Split | Origin days | Samples |
|---|---|---|
| Train | 1–6 September | 571 |
| Validation | 7–8 September | 188 |
| Test | 9–10 September | 188 |

Training and validation samples whose target time falls in the next period are removed.

Observed cold-cloud fraction (C13 < 235 K) at target time: validation 8.1%, test 4.2%.

## Known issues

- The window is mostly ocean with some land (Florida, Cuba, Hispaniola). September is within the Atlantic hurricane season; tropical systems may dominate some days. Not analysed per day.
- Pixels outside the Earth disk or flagged missing are NaN. In the one full CONUS scene inspected, 98.7% of pixels were valid; the missing share inside the cropped window was not measured per scene.
- Brightness temperature below 235 K includes thick cirrus and frontal cloud, not only convection.
- Float16 storage (see above).

## Licence and attribution

GOES-R series data are open data provided by NOAA. The store itself is not redistributed in this repository; it is rebuilt with the ingest command. For EUMETSAT data, the EUMETSAT Data Policy applies, including attribution ("Contains modified EUMETSAT data") and the restriction on redistributing the original numerical data of Recommended data.
