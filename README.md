# Regime-Aware AI Post-Processing of Monsoon Rainfall Forecasts

Improves a raw NWP rainfall forecast over India by detecting the prevailing
monsoon regime first, then applying a regime-specific bias correction, then
estimating a calibrated heavy-rainfall probability, then aggregating to district
level, then verifying the result against four baselines.

```
RAW NWP / AI FORECAST
      |
MULTIMODAL FEATURE ENGINEERING        features/engineering.py
(rainfall + humidity/wind/pressure/CAPE/vorticity + terrain + coast + month + lead)
      |
REGIME ENGINE                          models/regime_classifier.py
(rule labels -> XGBoost/CatBoost classifier -> regime probability + SHAP)
      |
REGIME ROUTER
(soft: blends all five regimes by probability; hard routing still available)
      |
REGIME-SPECIFIC RESIDUAL CORRECTION    models/bias_correction.py
(bias = observed - raw;  corrected = raw + predicted_bias;  one model per regime)
      |
PREDICTION INTERVALS                   models/uncertainty.py
(per-regime quantile regressors -> an honest range, coverage measured)
      |
HEAVY-RAIN PROBABILITY                 models/heavy_rain_probability.py
(LightGBM threshold classifiers -> isotonic / Platt calibration)
      |
DISTRICT PRODUCT                       aggregation/district.py
(GeoPandas zonal statistics -> rainfall + probability + warning level)
      |
VERIFICATION                           verification/metrics.py, report.py
(RMSE, bias, correlation, POD, FAR, CSI, ETS, FSS -- vs baselines)
      |
DASHBOARD                              api/static/
(three-panel maps, warning, both SHAP panels, reliability curve)
```

---

## Status: no data connected yet

**This repository contains no data and generates none.** Every loader reads a
local file you supply and raises a descriptive error when that file is absent.
The only fabricated rows in the codebase are 8 hand-written ones in
`data/loaders.py::_dummy_dataframe`, used exclusively by the test suite to check
that the stages hand off to each other correctly. They are never used as a
fallback, never tuned on, and no number derived from them means anything.

Consequently **there are no results in this repository yet**, and no accuracy or
improvement figure appears anywhere in the code, the comments or these docs.
Every number in the verification report is produced by running the metrics over
your real held-out data.

---

## (a) What data you need to add, and where

Full column-by-column contract: **[`rainfall_pipeline/data/README.md`](rainfall_pipeline/data/README.md)**.

Four inputs, all local files:

| What | Default path | Env override |
|---|---|---|
| Atmospheric predictors (ERA5 or equivalent) | `data_store/era5.parquet` | `RAINFALL_ERA5_PATH` |
| Observed rainfall (ground truth) | `data_store/observed_rainfall.parquet` | `RAINFALL_OBSERVED_PATH` |
| Raw NWP / AI forecast | `data_store/raw_nwp_forecast.parquet` | `RAINFALL_NWP_PATH` |
| District boundary polygons | `data_store/districts.geojson` | `RAINFALL_DISTRICTS_PATH` |

Set `RAINFALL_DATA_DIR` to relocate all four at once.

The three tables must share one grid and one 24-hour convention, with one row
per `(date, lat, lon)`. The single most common failure is a silent join collapse
from mismatched coordinate rounding or 0–360 longitudes — the data README covers
both.

The district shapefile is optional: if one of your tables already carries a
`district` column, the pipeline uses that instead.

---

## (b) How to run training end to end

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Check your files parse and join before committing to a full run:

```bash
python -c "from rainfall_pipeline.data.store import build_analysis_table as b; t=b(); print(len(t), t.date.min(), t.date.max())"
```

Then run everything — data ingest, chronological split, climatology, regime
classifier, all correctors, the probability head, and the verification report:

```bash
python -m rainfall_pipeline.training.run_full_training_pipeline
```

Useful flags:

| Flag | Effect |
|---|---|
| `--train-end 2021-09-30 --val-end 2022-09-30` | Explicit split boundaries. **Set these** — see the note on splitting below. |
| `--rebuild` | Re-read the raw files instead of the cached analysis table. |
| `--backend catboost` | Use CatBoost instead of XGBoost for the regime and correction models. |
| `--calibration sigmoid` | Platt scaling instead of isotonic regression. |
| `--min-rows-per-regime 500` | Raise the bar before a regime gets its own correction model. |
| `--artifact-dir path/` | Write models and the report somewhere other than `artifacts/`. |

Each stage also runs standalone, in this order:

```bash
python -m rainfall_pipeline.training.train_regime_classifier
python -m rainfall_pipeline.training.train_bias_correction
python -m rainfall_pipeline.training.train_heavy_rain_models
```

**Outputs** land in `artifacts/`:

```
climatology.parquet                      per-(lat,lon,month) rainfall/pressure climatology
regime_classifier.joblib                 the regime engine
baseline_b_global_ml.joblib              Baseline B
baseline_c_quantile_mapping.joblib       Baseline C
baseline_c_regime_quantile_mapping.joblib  per-regime QM ablation
bias_correction.joblib                   Model D (all per-regime models)
heavy_rain_probability.joblib            the calibrated probability head
prediction_intervals.joblib              per-regime quantile models for the range
training_manifest.json                   what was trained, when, on how many rows
verification_report.{json,md,html}       the five-model comparison
```

### On regime routing

The classifier emits a probability over all five regimes. Sending each row to
the `argmax` and discarding the rest throws away most of what it knows, and it
makes the correction discontinuous at the point where the leading regime flips
— which is exactly the Western Ghats, where a cell is genuinely part Coastal
and part Orographic.

So the default is **soft routing**:

```
predicted_bias = sum over regimes of  P(regime = r) * bias_model_r(row)
```

Probability mass belonging to a regime too thin to have earned its own model is
pooled onto the global fallback, so the weights always sum to one. A one-hot
distribution reproduces hard routing exactly — soft routing is a strict
generalisation, not a different model, and a test asserts that. Hard routing
remains available (`soft_routing=false`) for comparison.

Cost: soft routing evaluates up to one model per regime over the frame where
hard routing evaluates one model per row. Irrelevant for a single request,
roughly `n_regimes` times the work over a full verification set.

### On splitting

The split is **always chronological** — `verification/splits.py` has no
`random_state` and no `shuffle`, and a test asserts they never appear. Rainfall
is strongly autocorrelated in space and time; a shuffled split would leak
neighbouring days into training and turn every skill number into fiction.

If you do not pass `--train-end`/`--val-end`, the code falls back to splitting
70/15/15 on *unique dates*. That is a convenience for getting a run going. For
numbers you intend to present, set the boundaries explicitly (or in
`config/thresholds.py::SPLIT`) so the test set is a whole held-out monsoon
season.

---

## (c) How to start the API

```bash
uvicorn rainfall_pipeline.api.main:app --reload
```

Then open <http://127.0.0.1:8000/docs>.

| Endpoint | Purpose |
|---|---|
| `GET /` | **The dashboard** |
| `GET /predict?date=YYYY-MM-DD&district=<name>` | Full pipeline for one district: regime blend, regime probabilities, raw and corrected rainfall, calibrated heavy-rain probability per threshold, warning level, and SHAP explanations for *both* the regime call and the correction |
| `GET /grid?date=YYYY-MM-DD` | Gridded raw / corrected / difference / observed fields for one date, aligned on one lattice so they can share a legend |
| `GET /districts` | District names the service can serve |
| `GET /dates` | The date range the connected dataset covers |
| `GET /verification-report` | The saved report JSON |
| `GET /health` | What is loaded and what is still missing |
| `GET /timeline?date=&district=` | Regime and rainfall evolution across a window of days, with transitions flagged |
| `GET /risk-matrix?date=` | Every district scored and ranked for one date |
| `GET /watch?date=` | The highest-risk districts, as an early-warning view |
| `GET /events` | Biggest observed rainfall days, with the full forecast chain for replay |
| `GET /what-if?date=&district=` | Re-run the pipeline with humidity / wind / pressure / instability nudged |
| `GET /drivers?date=` | What is driving the corrections across the whole domain |
| `GET /api` | JSON index of the above |

`/predict` and `/grid` take `soft_routing=false` to fall back to routing every
row to its single most likely regime. `/predict` also takes an optional
`threshold=<mm>` for a caller's own rainfall limit; its probability is
interpolated between the trained thresholds and flagged `interpolated: true`,
because an interpolation is not a fitted model and must not be shown as one.

All artifacts load **once at startup**, not per request. The service starts
cleanly in every state: with no data and no models it comes up, `/health`
enumerates what is missing, and `/predict` returns `503` with the exact commands
to run — never a stack trace. Restart the API after training so it picks up the
new artifacts.

### The dashboard

Open <http://127.0.0.1:8000/>.

The dashboard is written for someone who has never heard of a monsoon regime or
a fractions skill score. The pipeline's own vocabulary — column names,
verification acronyms — never reaches the screen unless the reader asks for it:
`Orographic` is shown as "Hill rain", `cape_humidity` as "Unstable air combined
with moisture", RMSE as "typical error". A test asserts that no acronym leaks
into the default view, and two more assert that every feature, regime and model
in the pipeline has a plain-language name, so the vocabulary cannot silently
fall behind the code.

The technical scorecard is still one click away under *Show the full
scorecard*, with a glossary mapping each plain label back to its real name.

It shows, for one district on one date:

* a **headline sentence** carrying the whole result — what the weather model
  said, what the system changed it to, and whether that helped — over three
  large numbers on one shared scale;
* the **weather type**, named as a blend when a second type carries real
  probability ("Hill rain + Coastal rain"), each with a one-line description of
  the physical situation;
* **calibrated probabilities** at all three IMD thresholds, and a warning level
  written as a sentence rather than a bare label;
* **three map panels** — raw, corrected, and the correction applied — on one
  date, one extent and one shared legend, with the observed panel available as
  a fourth. Colour is assigned by IMD rainfall band, not linearly, because a
  linear scale washes out everything below 60 mm;
* **two SHAP panels**: *why did AI correct this?* (contributions in mm to the
  correction, at the district's wettest cell) and *why this regime?* These
  answer different questions and neither substitutes for the other;
* a **ranked comparison** of all six methods, best first, each with a plain
  description of why it is in the test and a bar showing how much of the
  weather model's error it removes — plus an automatic caveat when the no-ML
  ablation beats the ML model on spatial skill;
* a **plain verdict on whether the probabilities can be trusted**, reported
  from the worst-calibrated bin the system actually committed to rather than an
  average that the tens of thousands of near-zero forecasts would flatter.

### On the prediction interval

The corrected amount is reported as a range, from per-regime quantile
regressors fitted on the same residual target the point corrector uses
(`models/uncertainty.py`). They are fitted with a pinball loss rather than
derived from residual spread, because rainfall bias is strongly skewed and
heteroscedastic: a symmetric band around the point forecast would be far too
wide on dry days and far too narrow on the heavy ones that matter.

These are quantiles of the training residual distribution, **not calibrated
predictive intervals**. Whether observations actually land inside the band is an
empirical question, so the training run measures it on held-out data and records
both numbers in the manifest and the report:

```
Prediction intervals: nominal 80%, actual coverage 73.1% (mean width 9.8 mm)
```

The API returns both, and the dashboard says "this range is not well
calibrated" when they diverge by more than 10 points. A nominal 80% band
containing 55% of observations is a false reassurance, not a range.

### The dashboard's other views

Beyond the district forecast, the navigation carries:

* **Extreme rain watch** — every district scored and ranked for the day, with
  the district average *and* the wettest grid square shown side by side. A
  district can average a harmless 15 mm while one valley inside it takes 90 mm,
  and it is the valley that floods.
* **Past events** — the biggest observed rainfall days with the whole chain
  (weather model → our correction → what fell). Days inside the training period
  are labelled, because getting those right proves nothing.
* **How good is it?** — the ranked comparison, plus a breakdown that splits
  every score by weather type, region, district, rainfall intensity and lead
  time, with the best value per column picked out.
* **What-if simulator** — sliders for moisture, wind, pressure and instability
  that re-run the pipeline with the inputs nudged. Derived features are rebuilt
  from the adjusted columns, so raising the wind also moves wind speed, moisture
  flux and onshore flow rather than leaving the model reading a physically
  incoherent row. Every response carries a disclaimer: it is a probe of the
  model, not a forecast.

**"How to read this"** (top right) turns on contextual guidance beside every
panel: what it is, how to read it, and what to watch out for. Part of it is
static, and part is computed from the numbers currently on screen — it will
tell you that a 67/33 regime split means a transition zone, that the SHAP panel
is explaining the wettest cell rather than the district mean shown above, that
the D-vs-B margin is 1.7% and therefore not decisive, or that a reliability bin
is resting on a single event. Those are the things a reader misreads, and they
are only visible once the numbers exist. The setting is remembered per browser.

The dashboard holds no rainfall logic — it renders what the API returns, so
there is no second place for a number to be invented. When it is serving the
synthetic demo dataset it says so in a banner across the top, driven by a
marker file rather than by directory name.

---

## The five models the report compares

| ID | What it is | Why it is there |
|---|---|---|
| **A** | Raw NWP, uncorrected | The operational number to beat. A correction that cannot beat it is not worth deploying. |
| **B** | One global ML residual model, regime-blind | Sees the same features as D. The only difference is routing, so B-vs-D isolates the value of regime awareness. |
| **C** | Global empirical quantile mapping, no ML | Corrects the distribution but not individual events. C-vs-B isolates the value of machine learning. |
| **D** | Regime-specific residual correction | The core proposal. |
| **E** | D + calibrated heavy-rain probability | The full system. |

A per-regime quantile-mapping ablation is also fitted and reported, which
separates "regime awareness helps" from "machine learning helps" more cleanly
than any single pair.

D and E produce the **same rainfall field** — the probability head does not
change the corrected amount — so their continuous and rainfall-categorical
scores are identical by construction. The report states this in its own notes
rather than letting two identical rows be misread as a reproduced result. E is
distinguished by the probabilistic scores only it can produce.

### What the report contains

Per model: RMSE, mean bias, MAE, correlation; POD / FAR / CSI / ETS /
frequency bias and the raw contingency counts at all three IMD thresholds; and
FSS at neighbourhood widths of 1, 3 and 5 grid cells. All of it is broken down
by regime, by region, by district, by observed-intensity bucket **and by lead
time**, not just as one overall number. The lead-time breakdown appears only
when your forecast table actually spans several leads — with a single lead it
would be a one-row table that says nothing, so it is skipped rather than
padded. For Model E there is also a Brier score, a Brier skill score
against climatology, and a reliability table — read that table before trusting
any probability the system emits.

Undefined metrics (a threshold that was never exceeded, an FSS on a day with no
exceedance anywhere) serialise as `null`, never as a flattering default.

---

## Trying it without real data (synthetic demo)

There is a synthetic-data path for checking the machinery works before your real
files arrive:

```bash
./tools/run_demo.sh
```

That fabricates a 1-degree grid over 8-28N / 70-88E across three JJAS seasons
(146,034 rows), trains on 2020, calibrates on 2021 and verifies on a fully
held-out 2022, then writes `sample_artifacts/verification_report.md`.

`tools/make_synthetic_dataset.py` deliberately injects a **mechanism-dependent**
error into the fake forecast -- a severe dry bias over orography, a milder one
at the coast, a wet bias around depressions, spurious drizzle during breaks --
because a forecast whose error is pure noise cannot be corrected by anything and
would make the whole exercise vacuous.

**Everything this produces is fake.** It is not ERA5, not IMD, not any real
forecast model, and no number from it may be presented as a result. It lives in
`sample_data/` and `sample_artifacts/`, never in `data_store/` or `artifacts/`,
and the generator refuses to write into `data_store/`. Both directories are
gitignored.

## Configuration

| File | Holds |
|---|---|
| `config/thresholds.py` | IMD rainfall categories, regime labels, **the rule-based regime criteria**, warning-level cut-points, FSS windows, intensity buckets, split boundaries |
| `config/regions.py` | Bounding boxes, sub-regions, file paths, grid spacing, CRS |

The regime rules in `RegimeRuleConfig` are **simplified placeholders** for the
published active/break criteria — plausible thresholds on core-zone rainfall
anomaly and 850 hPa zonal wind, plus vorticity/pressure, terrain and coastal
tests. Replace them with the official IMD/IITM definitions when you have them;
every consumer reads from that one dataclass, so nothing else changes.

Similarly, the warning-level probability cut-points (0.30 / 0.50 / 0.50) are
untuned starting points, not calibrated decision thresholds. Revisit them once
you can see the real base rates in the report.

---

## Running the tests

```bash
pytest
```

The suite covers the data contract and loader errors, feature engineering
(including explicit leakage checks), each model's fit/predict/save/load and
routing (hard *and* soft, including that soft reproduces hard exactly on a
one-hot distribution), both SHAP explainers, the district aggregation rules,
every metric against hand-computed values, the chronological splitter, the
lead-time stratification, the report renderer, the API in both cold and trained
states — including the gridded map endpoint and the regime-blend labelling —
the static-field builder, and one end-to-end run of the full training
pipeline.

It asserts on **structure and contracts only**. There is no accuracy assertion
anywhere, because on 8 fabricated rows no accuracy assertion would mean
anything.

---

## Layout

```
rainfall_pipeline/
  config/       thresholds.py  regions.py
  data/         loaders.py  schema.py  store.py  README.md
  features/     engineering.py
  models/       regime_classifier.py  bias_correction.py
                heavy_rain_probability.py  baselines.py  uncertainty.py
  aggregation/  district.py
  verification/ metrics.py  splits.py  report.py
  api/          main.py  schemas.py  static/ (dashboard)
  training/     common.py  train_regime_classifier.py  train_bias_correction.py
                train_heavy_rain_models.py  run_full_training_pipeline.py
  tests/        test_data.py  test_features.py  test_models.py
                test_verification.py  test_api.py  test_training.py
                test_tools.py  test_uncertainty.py
tools/          make_synthetic_dataset.py  build_static_fields.py  run_demo.sh
data_store/     >>> your input files go here <<<
artifacts/      trained models + verification report (produced by training)
```

---

## Known limitations to be aware of before presenting this

* **The regime definitions are placeholders.** They are simplified stand-ins,
  not the official IMD/IITM criteria. Say so rather than implying otherwise.
* **Regime labels are rules, not truth.** The classifier's agreement with them
  measures whether it distilled the rule, not whether the rule is right. The
  training script reports that agreement for exactly this reason.
* **Training uses rule labels; prediction uses classifier labels.** The rules
  need the observation, which does not exist at forecast time. The corrector
  therefore sees a slightly different regime mix in production than in training.
  Soft routing softens this — a misrouted row is now blended rather than sent
  wholly to the wrong model — but it does not remove it. Each per-regime model
  is still *fitted* on hard rule labels.
* **You must supply `elevation` and `coastal_distance` yourself.** They are
  static per-cell fields the pipeline reads but cannot derive. Build them once
  with `tools/build_static_fields.py`; without them every orographic and
  coastal feature is dead weight, and two of the five regimes become
  unreachable.
* **Rare thresholds may be unfittable.** 204.4 mm exceedances are genuinely
  rare; if a short training period contains none, that classifier falls back to
  a constant base rate and the report flags it as degenerate.
* **The district join uses cell centres.** Fine for a 0.25° grid over India; if
  you move to a much coarser grid, replace it with an area-weighted overlay.
* **Calibration quality depends on the validation split.** With no validation
  rows the calibrator is fitted on training predictions, and the report says so
  — the reliability table is then optimistic.
