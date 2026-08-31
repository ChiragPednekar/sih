# Data contract — what you need to prepare

Nothing in this repository downloads data, and nothing generates synthetic data
as a fallback. Every loader reads a **local file you supply**. Until those files
exist, each loader raises an error that repeats the requirements below.

There are four inputs. Three are flat tables; one is a polygon layer.

| # | What | Loader | Default path | Env override |
|---|------|--------|--------------|--------------|
| 1 | Atmospheric predictors | `load_era5` | `data_store/era5.parquet` | `RAINFALL_ERA5_PATH` |
| 2 | Observed rainfall (truth) | `load_observed_rainfall` | `data_store/observed_rainfall.parquet` | `RAINFALL_OBSERVED_PATH` |
| 3 | Raw NWP / AI forecast | `load_raw_nwp_forecast` | `data_store/raw_nwp_forecast.parquet` | `RAINFALL_NWP_PATH` |
| 4 | District boundaries | `load_district_boundaries` | `data_store/districts.geojson` | `RAINFALL_DISTRICTS_PATH` |

Set `RAINFALL_DATA_DIR` to move all four defaults at once.

---

## Which real product fills each slot

The loaders are deliberately **source-agnostic** — they read a schema, not a
named product — so the pipeline works with GFS or IFS or an AI forecast without
a code change. That is a design choice, but it means the product names from the
proposal do not appear anywhere in the code. This table is the mapping.

| Slot | Proposal names | Where to get it | Notes |
|---|---|---|---|
| `era5.parquet` | **ERA5** | [Copernicus CDS](https://cds.climate.copernicus.eu) — `reanalysis-era5-single-levels` and `-pressure-levels` | Needs a free CDS account and `cdsapi`. Pressure-level request for `u`/`v` at 850 and 200 hPa; single-level for MSL, CAPE, TCWV, OLR. |
| `observed_rainfall.parquet` | **IMERG Final** | [GES DISC](https://disc.gsfc.nasa.gov) — `GPM_3IMERGDF` v07 | Daily, 0.1°. Regrid to your target grid. See the caveat below. |
| `raw_nwp_forecast.parquet` | **GFS 0.25°** | [NOAA NOMADS](https://nomads.ncep.noaa.gov) or the [AWS open-data mirror](https://registry.opendata.aws/noaa-gfs-bdp-pds/) | `APCP` accumulated precipitation. Accumulate to the same 24-h window as the observation — this is the single most common source of a silent bias. |
| `districts.geojson` | **District GIS** | [Survey of India](https://surveyofindia.gov.in) / [data.gov.in](https://data.gov.in) | Any polygon layer with a district-name field. Set `RAINFALL_DISTRICT_NAME_FIELD` if the column is not `district`. |
| `elevation`, `coastal_distance` columns | **SRTM** | [USGS EarthExplorer](https://earthexplorer.usgs.gov) SRTM 1 Arc-Second, or [NASADEM](https://lpdaac.usgs.gov) | Static fields — derive once and merge into the ERA5 table. See the section below. |

### Choosing the observation: IMERG or IMD gauge

The pipeline classifies rainfall with the **IMD daily categories** (64.5 /
115.6 / 204.4 mm). If you train against IMERG, you are applying gauge-derived
category boundaries to a satellite product, and the two disagree most exactly
where this project claims its value: **IMERG systematically underestimates
heavy orographic rainfall over the Western Ghats.** A heavy-rain classifier
trained on it will under-forecast the events the Orographic regime exists to
capture.

Preferences, best first:

1. **IMD gridded gauge rainfall at 0.25°** (IMD Pune). Same source as the
   category definitions, so the thresholds mean what they say.
2. **IMERG Final**, stating the choice in the report and expecting the
   orographic heavy-rain scores to be pessimistic.
3. A blend, gap-filling gauge with satellite — more work, and it needs its own
   verification before it can be trusted.

Whichever you pick, say so in the report. A reviewer from IMD will ask.

### Deriving the static fields

`elevation` and `coastal_distance` are **static** — they depend on the grid,
not the date — so they are computed once and merged into the ERA5 table rather
than being loaded per run:

```bash
python -m tools.build_static_fields --grid data_store/era5.parquet --dem srtm_india.tif --coastline coastline.geojson --out data_store/static_fields.parquet
```

`coastal_distance` needs only a coastline vector layer (geopandas, already a
dependency). `elevation` needs a DEM raster and therefore `rasterio`, which is
an optional install — the script tells you if it is missing.

---

## Universal rules for the three tables

* **Format**: `.parquet` (preferred) or `.csv`.
* **Granularity**: exactly one row per `(date, lat, lon)`. No duplicates —
  duplicates silently multiply rows in the join.
* **`date`**: one row per day, any pandas-parseable format (`YYYY-MM-DD` is
  safest). Parsed and normalised to midnight on load.
* **`lat`**: decimal degrees north.
* **`lon`**: decimal degrees east, in the **−180…180** convention. ERA5 ships
  0…360 longitudes — convert them, or the spatial filters and the district join
  will silently drop everything. `loaders.flatten_gridded_dataset` does this
  conversion for you.
* **Grid alignment**: all three tables must sit on the **same grid**. The join
  is on exact float equality, so `19.25` and `19.250001` are different cells.
  Round every coordinate to the same precision before saving, e.g.
  `df["lat"] = df["lat"].round(2)`.
* **Time alignment**: the forecast valid window and the observation
  accumulation window must be the same 24 hours. IMD daily rainfall runs
  0830 IST → 0830 IST; if your NWP output is 00 UTC → 00 UTC, shift one of them
  before saving rather than letting the mismatch quietly degrade every model.
* **Units**: as listed below. Rainfall in **mm**, not metres — a factor of 1000
  will not be detected for you.
* **Missing values**: leave them as NaN/null. Do not fill them with zeros; a
  zero rainfall observation and a missing one mean very different things.

---

## 1. ERA5 atmospheric predictors — `load_era5`

**Required columns**

| Column | Units | Notes |
|---|---|---|
| `date` | — | Valid date |
| `lat` | °N | Grid cell centre |
| `lon` | °E | Grid cell centre, −180…180 |
| `pressure_msl` | hPa | Mean sea level pressure |
| `wind_u_850` | m/s | Zonal wind at 850 hPa |
| `wind_v_850` | m/s | Meridional wind at 850 hPa |
| `wind_u_200` | m/s | Zonal wind at 200 hPa |
| `wind_v_200` | m/s | Meridional wind at 200 hPa |
| `olr` | W/m² | Outgoing longwave radiation |
| `humidity` | % (0–100) | Relative humidity at one representative level, e.g. 700 hPa |
| `cape` | J/kg | Convective available potential energy |
| `vorticity` | 1/s | Relative vorticity at 850 hPa (typically order 1e−5) |

**Optional but strongly recommended** (they must reach the pipeline from
*somewhere*; putting them here is easiest, since they are static per grid cell):

| Column | Units | Notes |
|---|---|---|
| `elevation` | m | Terrain height. Drives the Orographic regime. |
| `coastal_distance` | km | Distance to nearest coastline; 0 on the coast. Drives the Coastal regime. |
| `district` | — | District name. Optional if you supply the shapefile instead. |

Without `elevation` and `coastal_distance` the Orographic and Coastal regimes
can never be assigned, and the five-regime router collapses to three.

**Example header row (CSV)**

```csv
date,lat,lon,pressure_msl,wind_u_850,wind_v_850,wind_u_200,wind_v_200,olr,humidity,cape,vorticity,elevation,coastal_distance
2020-07-01,19.00,73.00,1002.0,12.0,4.0,-18.0,2.0,190.0,88.0,1400.0,0.00005,560.0,60.0
```

### If your ERA5 data is NetCDF or GRIB

Convert it once, offline, and save the result as Parquet:

```python
import xarray as xr
from rainfall_pipeline.data.loaders import flatten_gridded_dataset

ds = xr.open_dataset("era5_july_2020.nc")          # your file, your machine
df = flatten_gridded_dataset(
    ds,
    variable_map={
        "msl": "pressure_msl",
        "u": "wind_u_850",
        "v": "wind_v_850",
        "olr": "olr",
        "r": "humidity",
        "cape": "cape",
        "vo": "vorticity",
    },
    time_dim="time", lat_dim="latitude", lon_dim="longitude",
)
df.to_parquet("data_store/era5.parquet", index=False)
```

Daily fields from sub-daily data: resample first (`ds.resample(time="1D").mean()`
for state variables, `.sum()` for accumulations).

---

## 2. Observed rainfall — `load_observed_rainfall`

This is the ground truth. Everything the project claims rests on it.

**Required columns**

| Column | Units | Notes |
|---|---|---|
| `date` | — | Valid date |
| `lat` | °N | Must match the ERA5 grid exactly |
| `lon` | °E | Must match the ERA5 grid exactly |
| `observed_mm` | mm | 24 h accumulated rainfall |

**Optional**: `district`.

```csv
date,lat,lon,observed_mm
2020-07-01,19.00,73.00,72.0
```

If your source is station data rather than a grid, interpolate it to the model
grid *before* saving — the pipeline does no spatial interpolation, and doing it
here keeps the interpolation choice visible and yours.

---

## 3. Raw NWP forecast — `load_raw_nwp_forecast`

The forecast being corrected. It must be a **genuine forecast**, not an
analysis: if it has seen the observation it is trying to predict, every skill
number downstream is meaningless.

**Required columns**

| Column | Units | Notes |
|---|---|---|
| `date` | — | **Valid** date, not the initialisation date |
| `lat` | °N | Same grid |
| `lon` | °E | Same grid |
| `raw_forecast_mm` | mm | 24 h accumulated rainfall forecast |

**Optional**

| Column | Units | Notes |
|---|---|---|
| `lead_time` | days | Forecast lead. Defaults to 1 if absent. Include it if you mix leads — bias grows with lead time and the models can use it. |

```csv
date,lat,lon,raw_forecast_mm,lead_time
2020-07-01,19.00,73.00,40.0,1
```

If you keep several leads, `date` is the valid date and `lead_time` distinguishes
the rows — which makes `(date, lat, lon)` no longer unique. In that case, filter
to a single lead before saving, or extend the join keys in
`data/schema.py::KEY_COLUMNS`.

---

## 4. District boundaries — `load_district_boundaries`

**Format**: `.shp` (bring the `.dbf`/`.shx`/`.prj` siblings), `.gpkg`, or
`.geojson`.

**Requirements**

* Geometry: `Polygon` / `MultiPolygon`, one feature per district.
* A district-name attribute. The loader looks for `district`, then falls back to
  `DISTRICT`, `dtname`, `NAME_2`, `distname`. If yours is named something else,
  rename the column or set `RAINFALL_DISTRICT_NAME_FIELD`.
* Any CRS — it is reprojected to EPSG:4326 on load.

No shapefile is bundled and none is downloaded. Survey of India / data.gov.in
administrative boundaries and the GADM level-2 dataset are the usual sources;
check the licence of whichever you pick before using it in a submission.

**If you skip the shapefile**: put a `district` column in one of the three
tables instead. District aggregation and `/districts` will work from that.
Without either, the pipeline still trains and verifies at grid-cell level;
only the district product is unavailable.

---

## Verifying your files before training

```python
from rainfall_pipeline.data.store import build_analysis_table

table = build_analysis_table()          # reads all three, joins, caches
print(len(table), table["date"].min(), table["date"].max())
print(table[["raw_forecast_mm", "observed_mm"]].describe())
```

If the row count is far lower than you expect, the join is dropping cells —
almost always a grid-alignment or longitude-convention problem. See the
universal rules above.

The joined table is cached to `data_store/store/analysis_table.parquet`, so
later runs skip re-reading the raw files. Pass `--rebuild` to any training
script to force a re-read after you change an input.
