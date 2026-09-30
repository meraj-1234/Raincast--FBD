# PS26079 — AI-Based Forecast Bust Detection (Prototype)

Predicts **when an existing medium-range rainfall forecast is likely to fail**
over Indian subdivisions — which region, which lead day, and why — rather
than trying to forecast the weather itself. This is decision support: it
flags forecasts for a human forecaster to double-check, it never issues or
suppresses a warning itself.

## The five required outputs (mapped to this repo)

| PS26079 requirement          | Where it lives |
|-------------------------------|----------------|
| Forecast confidence map       | `ps26079_dashboard.html` — subdivision × Day 1–10 grid |
| Bust probability              | `src/train_model.py` → `bust_probability` column |
| Error-prone area detection    | `src/explain.py` → `outputs/review_queue.json` (ranked) |
| Explainable output            | `src/explain.py` — SHAP → plain-language reasons |
| Prototype dashboard/API       | `ps26079_dashboard.html` (self-contained, no server needed) |

## What's real vs. synthetic — read this first

This build environment has **no network route** to the data PS26079 actually
needs:
- IMD 0.25° gridded rainfall (imdpune.gov.in)
- NWP forecast archives / ensemble spread (WeatherBench2 on Google Cloud Storage)
- ERA5 reanalysis (also WeatherBench2)

So `src/generate_synthetic.py` generates data with the **same statistical
shape**: seasonal rainfall cycle, forecast error that grows with lead day,
and — the key realistic failure mode — ensemble spread that *collapses*
right before regime-driven busts (false confidence), which is exactly the
pattern a real "trust the spread" baseline gets wrong.

**Everything downstream of that one file is real, unmodified logic**: the
bust label, feature engineering, XGBoost model, isotonic calibration,
evaluation, and SHAP explainability. Swap `generate_synthetic.py` for a real
ingestion script with the same output schema (see the table in
`src/generate_synthetic.py`'s docstring) and the rest of the pipeline runs
unchanged.

## Bust definition

```
bust = |forecast − observed| >= max(10mm, P95(subdivision, month))   # magnitude
       AND category(forecast) != category(observed)                   # decision flips
       AND max(forecast, observed) >= 15.6 mm/day                     # significant rain
```
P95 threshold is fitted on **training years only** — see `src/labels.py`.

## Results on held-out year 2022 (synthetic data), Day 3–7 decision band

| Predictor                              | AUROC | Brier  |
|-----------------------------------------|-------|--------|
| Climatology                              | 0.643 | 0.0190 |
| Forecast amount alone                    | 0.798 | 0.0179 |
| Ensemble spread (operational baseline)   | 0.695 | 0.0189 |
| **XGBoost + isotonic calibration**       | **0.905** | **0.0159** |

Full numbers: `outputs/eval_results.json`.

## Run it yourself

```
pip install xgboost scikit-learn pandas numpy pyarrow shap
cd src
python generate_synthetic.py     # -> data/dataset_raw.parquet
python train_model.py            # -> models/model.pkl, outputs/eval_results.json
python explain.py                # -> outputs/review_queue.json
python build_dashboard_data.py   # -> outputs/dashboard_bundle.json
```
Then rebuild the dashboard by injecting `outputs/dashboard_bundle.json` into
`dashboard/template.html` in place of `__BUNDLE_JSON__`.

## Honest limitations of this prototype

- Trained and evaluated entirely on **synthetic** data — AUROC numbers are
  illustrative of the *approach*, not a claim about real forecast skill.
  With real IMD + WeatherBench2 data the numbers will differ.
  A similarly-scoped real-data build (same architecture, same labels)
  reports **AUROC ≈ 0.84** against a real-ensemble baseline of ≈0.79–0.81 —
  a modest, credible margin, not the huge gap synthetic data can produce.
- Only 12 subdivisions and 10 representative dashboard dates are included
  to keep the demo small — the pipeline itself scales to all 36 IMD
  subdivisions and every day of the year.
- No real out-of-distribution / staleness detection is implemented here
  (the real pipeline this is modeled on adds a Mahalanobis-distance refusal
  layer and a `mode=live` staleness check — worth adding before a judged
  demo if you present this as your own SIH submission).
- For a hackathon submission, be upfront with judges that the numbers are
  on synthetic data pending real IMD/NWP access, and show that the code
  path is ready to take real data unchanged.

## Repository layout

```
src/
  generate_synthetic.py   synthetic data stand-in (swap for real ingestion)
  labels.py                bust label definition
  features.py               causal feature engineering
  train_model.py            baselines + XGBoost + isotonic calibration + eval
  explain.py                 SHAP -> plain-language reasons
  build_dashboard_data.py    aggregates results into dashboard JSON
dashboard/
  template.html              dashboard HTML template
  index.html                  built dashboard (data injected)
data/        generated dataset
models/      trained model pickle
outputs/     eval results, review queue, dashboard bundle
```

## UPDATE — real IMD rainfall data integrated (2022–2025)

Real IMD 0.25° gridded rainfall NetCDFs (`RF25_ind<year>_rfp25.nc`, 2022-2025)
were provided and are now the ground-truth (`observed_mm`) side of the
pipeline:

- `src/ingest_real_imd.py` — aggregates the real grids to 12 subdivisions
  using **approximate lat/lon bounding boxes** (real IMD subdivision polygon
  shapefiles were not reachable from this network — see limitations below)
- `src/build_hybrid_dataset.py` — real observed rainfall + simulated forecast
  overlay (same error model as before, now grounded in real seasonality/climatology)
- `src/train_model_real.py`, `src/explain_real.py`, `src/build_dashboard_data_real.py`
  — same logic as the originals, trained/evaluated on the hybrid dataset,
  held out year **2025**, trained on **2022-2024**
- `ps26079_dashboard_real.html` — dashboard rebuilt on these results

### Still missing: real forecast / NWP data

The bust label needs `|forecast - observed|`. Only `observed` is real now.
`forecast_mm`, `ens_spread`, `jumpiness`, `moisture_anom`, `shear_anom` are
still simulated (same model as `generate_synthetic.py`, applied on top of
real rainfall instead of synthetic rainfall). Get real forecast/ensemble data
from:
- **ECMWF Open Data** (https://www.ecmwf.int/en/forecasts/datasets/open-data) — free, rolling recent dates only
- **NCMRWF** (https://www.ncmrwf.gov.in/) — India GFS, historical, registration required

Once you have that, replace `forecast_mm`/`ens_spread`/etc. in
`build_hybrid_dataset.py` with real values on the same schema — labels,
features, model and calibration code do not need to change.

### Results on held-out 2025 (real rainfall + simulated forecast), Day 3-7 band

| Predictor | AUROC |
|---|---|
| Climatology | 0.755 |
| Ensemble spread baseline | 0.819 |
| **XGBoost + calibration** | **0.975** |

(Numbers are higher than the fully-synthetic run because the regime flag here
is derived directly from real rainfall extremity, which the model can learn
more cleanly than the earlier fully-synthetic regime process — this is a
methodological artifact, not evidence the real system will hit 0.975.)

### Limitation: bounding-box subdivisions, not real polygons

`SUBDIVISION_BOXES` in `ingest_real_imd.py` are rectangular approximations of
each subdivision, not the true IMD polygon. Near a box edge, rainfall from a
neighbouring subdivision can leak in. If you can get real IMD subdivision
shapefiles (or district shapefiles + a district→subdivision mapping), replace
the box-based aggregation with exact polygon-overlap weights, as the
reference architecture in the original chat does with `datameet/maps`
district boundaries.

## UPDATE 2 — 36 subdivisions, geographic map, API (final)

- **36 IMD subdivisions** now used throughout (`SUBDIVISION_BOXES` in
  `src/ingest_real_imd.py`) instead of 12.
- **Real geographic map**: `ps26079_dashboard_FINAL.html` plots subdivisions
  by their real lat/lon centroid on an India-extent canvas (no shapefile
  needed since points are positioned by coordinates, not filled polygons).
  A second tab shows a **fully real, non-downscaled** grid-point snapshot
  (from real ECMWF + real IMD, 5 demo dates, `grid_confidence_table_FINAL.csv`)
  for comparison against the subdivision-level approximation.
- **API**: `app.py` — FastAPI app with `/confidence-map`, `/bust-probability`,
  `/review-queue`, `/error-prone-areas`, `/model-info`. Run locally:
  ```
  pip install fastapi uvicorn
  uvicorn app:app --reload
  ```
  Then open `http://127.0.0.1:8000/docs`.

### Final honest result (36 subdivisions, real ECMWF + real IMD)

Train: monsoon 2023. Test (held out): 21 Jul – 30 Sep 2024.

| Predictor | AUROC |
|---|---|
| Climatology | 0.469 |
| **Forecast amount alone** | **0.606** |
| Lagged-spread proxy | 0.521 |
| XGBoost + calibration | 0.590 |

**The model does not beat the simplest baseline (forecast amount alone).**
This is the honest real-data result and should be presented as such. Likely
causes, in order of expected impact:
1. No ERA5 (moisture flux, shear, jet indices) -- the features the reference
   architecture leans on most for regime detection are entirely absent here.
2. Only 2 years of real forecast data (2023 train / 2024 test) -- not enough
   for XGBoost to learn a generalizable pattern rather than 2023-specific noise.
3. India-mean forecast downscaled via climatology ratio, not a true per-
   subdivision NWP field -- adds noise that swamps whatever signal exists.

### Where to improve if you have more time before judging

Ranked by expected impact per hour of effort:
1. Get real ERA5 features (free via Copernicus CDS, registration required) --
   likely the single biggest lever, matching what the reference architecture
   in this chat relies on most.
2. Re-fetch ECMWF forecast keeping the full spatial grid per subdivision
   (scripts/fetch_real_ecmwf_LOCAL.py currently only pulls the India-mean;
   the grid-level fetch in scripts/fetch_grid_demo_LOCAL.py proves this works,
   just needs to run over the full training period, not 5 demo dates).
3. If judged before more data is feasible, present the **negative result
   honestly** alongside the reference architecture's real numbers (AUROC
   ~0.84 with ERA5 + real ensemble) as the target this prototype is working
   toward -- judges respond well to "here's what real data actually looks
   like, here's our plan to close the gap" over inflated synthetic numbers.

## UPDATE 3 — REAL ERA5 features integrated (final result)

Real ERA5 reanalysis (2023-2024 monsoon, user-fetched via Copernicus CDS)
integrated: dewpoint depression, 850hPa relative humidity, 850-200hPa wind
shear, mean sea level pressure tendency. Attached at forecast **init_date**
(not valid_date) -- causally correct, no leakage.

### FINAL honest result (36 subdivisions, real ECMWF + real IMD + real ERA5)

Train: monsoon 2023. Held-out test: 21 Jul - 30 Sep 2024.

| Predictor | AUROC | AUPRC | Brier | LogLoss |
|---|---|---|---|---|
| Climatology | 0.469 | 0.053 | 0.057 | 0.627 |
| Forecast amount alone (baseline) | 0.606 | - | - | - |
| XGBoost, forecast only | 0.590 | 0.072 | 0.057 | 0.257 |
| XGBoost, + climatology | 0.592 | 0.072 | 0.057 | 0.247 |
| **XGBoost, + real ERA5** | **0.617** | **0.077** | 0.057 | 0.240 |
| **Logistic Regression, ERA5 features (best)** | **0.619** | 0.078 | 0.057 | 0.240 |
| Random Forest, ERA5 features | 0.614 | 0.076 | 0.056 | 0.254 |

**This is the first configuration that genuinely beats the forecast-amount
baseline.** Improvement is modest (+0.011 to +0.013 AUROC) but real -- driven
entirely by real ERA5 moisture/shear/pressure data, confirming the diagnosis
made earlier in this project: the missing meteorological state was the
binding constraint, not model choice (Logistic Regression ~ Random Forest ~
XGBoost once ERA5 is added -- the signal is close to linear).

Ablation history (what was tried, in order):
1. Forecast + lagged-spread proxy + jumpiness: 0.590 (no better than baseline)
2. + historical error features (expanding MAE/bias): 0.564 (WORSE -- too
   little history, 1 year, to estimate stable expanding statistics)
3. + real ERA5 meteorological features: **0.617** (first real improvement)

### PS26079 18-phase spec -- status against each phase

| Phase | Status |
|---|---|
| 1. Audit existing project | Done informally across this chat; no separate audit doc |
| 2. Real data pipeline (Date/Lat/Lon/Lead/Forecast/Observed/Error) | Done at grid level for 5 demo dates (`grid_confidence_table_FINAL.csv`); subdivision level for the full 2023-2024 training period |
| 3. Spatial alignment (nearest-neighbour vs bilinear) | Nearest-neighbour used and documented (`build_grid_confidence_table.py`); bilinear not implemented -- nearest-neighbour is defensible given IMD/ECMWF grids differ by resolution across the 2024-02 cutover and a bilinear scheme would need to handle that discontinuity explicitly |
| 4. Expand to full 2023-2025 grid-level, chunked | NOT done -- only 5 demo dates at grid level. Full grid-level historical fetch was scoped out (would take many more hours of ECMWF Open Data downloads); training uses subdivision-mean instead |
| 5. Defensible bust label, prevalence reporting | Done -- P90/P95-based (`labels.py`), prevalence reported every training run (train ~3.3%, test ~5.9%) |
| 6. Historical error features, leak-safe | Done (`historical_features.py`, expanding-window, shift(1) before expanding) -- but empirically HURT performance with only 1 year of history; kept in codebase, excluded from the final feature set |
| 7. Real ENS ensemble | NOT available -- no real 50-member ensemble fetched (would be ~50x the ECMWF Open Data requests already made). `lagged_spread` is explicitly labeled as a proxy, never called "ensemble" |
| 8. Real meteorological features (ERA5) | Done -- dewpoint depression, 850hPa RH, 850-200hPa wind shear, MSLP tendency, all real |
| 9. Spatial neighbourhood features (3x3, gradients) | NOT done at scale -- would need the full grid-level fetch (phase 4); demonstrated only on the 5 grid demo dates |
| 10. Model comparison (LogReg/RF/XGBoost/LightGBM/CatBoost) | LogReg, Random Forest, XGBoost compared on identical splits (`train_with_era5.py`). LightGBM/CatBoost not tried -- unlikely to change the conclusion since LogReg already matches XGBoost |
| 11. Calibration (Platt vs isotonic) + AUPRC/Brier/LogLoss | Isotonic used throughout; Platt not compared. AUPRC, Brier, LogLoss all reported (this update) |
| 12. Ablation study | Done -- 3-stage ablation reported above, run on real held-out data |
| 13. Explainability grounded in real features | Done -- SHAP + plain-language reasons now reference real ERA5 variables (e.g. "sea-level pressure dropping fast -- deepening system", "850hPa humidity unusually high") |
| 14. Confidence score (1 - P_bust) | Done, documented in API `/confidence-map` |
| 15. Geographic confidence map, Day1-10, real lat/lon | Done -- `ps26079_dashboard_ERA5_FINAL.html`, subdivision centroids on real coordinates + a real fine-grid snapshot tab for 5 dates |
| 16. Dashboard/API | Done -- dashboard above; `app.py` FastAPI with 5 endpoints |
| 17. Automated tests | NOT done -- no test suite written (time-boxed out; would cover unit conversion, date matching, leakage checks as specified) |
| 18. Final evaluation report | This README section + `outputs/era5_results.json` |

### What is genuinely complete
Real observed rainfall (IMD, 4 years), real forecast (ECMWF, 2 years), real
ERA5 meteorological features, leak-safe historical-error code (even though
not in the final feature set), honest ablation + model comparison,
explainability grounded in real variables, geographic dashboard, API.

### What still needs more real data or time
Real ENS ensemble, full grid-level (not subdivision-mean) training data,
spatial neighbourhood features at scale, LightGBM/CatBoost, Platt-vs-isotonic
comparison, automated test suite, 2025 forecast data (to get an independent
third test year rather than a single 2023-train/2024-test split).

### Commands
```
# rebuild everything from scratch (needs the real data files documented above)
cd src
python ingest_real_imd.py
python build_real_forecast_dataset.py
python extract_era5_features.py          # needs data/era5/*.grib
python train_with_era5.py                # final model
python explain_era5.py
python build_dashboard_data_era5.py

# API
cd .. && uvicorn app:app --reload
```
