# Dataset card: cropped brightness temperature, GOES-19 ABI and Meteosat SEVIRI, 1–10 September 2026

## Summary

Two time-regular Zarr stores of brightness temperature at two infrared channels over a fixed 1024 × 1024 pixel window, built by `python -m eo_ml_ready.ingest` (`--source goes`, the default, or `--source seviri`). The two stores share layout, period, cadence and splits. The first part of this card describes the GOES store; the SEVIRI store follows after "Known issues".

## Source (GOES)

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

## SEVIRI store

### Source

- Satellite and product: Meteosat SEVIRI Level 1.5 full-disk imagery (collection `EO:EUM:DAT:MSG:HRSEVIRI`, native format) from the EUMETSAT Data Store, retrieved with `eumdac`. Account and licence needed; see below.
- Channels: IR 10.8 µm (window; counterpart of GOES C13) and WV 6.2 µm (upper-level water vapour; counterpart of C08), calibrated to brightness temperature in kelvin by satpy (`seviri_l1b_native`).
- Cadence 15 minutes (one scene per time step); pixel size about 3 km at the sub-satellite point.
- The native files are stored in scan order (south to north, east to west). The reader flips them to north-up and west-to-east before cropping, and takes the corner coordinates from the satpy area definition.

### Contents

| Item | Value |
|---|---|
| Variable | `bt` (time, channel, y, x), float16, kelvin, NaN where missing |
| Period | 2026-09-01 00:00 to 2026-09-10 23:45 UTC |
| Time steps | 960 |
| Window | rows 960–1984, columns 1400–2424 of the 3712 × 3712 north-up full disk |
| Corners (lat, lon) | NW 25.8, −14.1; NE 25.9, 17.7; SW −3.5, −12.5; SE −3.5, 15.6 |
| Coordinates | `time` (UTC), `channel`, `y`, `x` (projection coordinates in metres) |
| Attributes | title, source, channels, units, window pixels, corner latitudes and longitudes |
| Chunks | (1, 2, 256, 256) |

Values outside 100–350 K are set to NaN by the reader (needed for zero radiances); the first day was ingested before this rule was added, and it contained no such pixels.

### Checks

`scripts/check_store.py` on the full store: IR 10.8 µm between 179.1 and 333.5 K, WV 6.2 µm between 180.4 and 258.5 K, no infinite and no missing pixels, no frame with more than 1% missing, no near-identical consecutive frames (mean absolute change of the window channel 2.59 K per step). Mean cold-cloud fraction (IR 10.8 µm < 235 K) 3.95%, at most 9.15% in a frame. The orientation was checked on a plotted frame and against the corner coordinates.

### Samples and splits

Same rule and dates as for GOES (train 1–6 September, 571 samples; validation 7–8 September, 188; test 9–10 September, 188). Observed cold-cloud fraction at target time: validation 4.4%, test 3.4%.

### Known issues

- The window covers the Sahel, the Sahara edge, the Gulf of Guinea and northern Central Africa: mostly land, with strong daily cycles and large organised convective systems. Not analysed per day or by land and sea.
- Some of the downloaded files produce the satpy warning "The quality flag for this file indicates not OK". No frame is dropped and the store has no missing pixels, but these scenes were not analysed separately.
- Float16 storage (see above).
- The cold-cloud label is a brightness temperature threshold at 10.8 µm, which differs slightly in wavelength from the GOES 10.3 µm channel.

## Licence and attribution

GOES-R series data are open data provided by NOAA. The SEVIRI data come from EUMETSAT and are subject to the EUMETSAT Data Policy and the licence of the collection used; check the current terms before use. Attribution: "Contains modified EUMETSAT data". The stores themselves are not redistributed in this repository; they are rebuilt with the ingest command (the SEVIRI download needs your own EUMETSAT account and licence).
