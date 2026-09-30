# PS26079 — Forecast Bust Detection (React + Tailwind + Node/Express)

Interactive dashboard + JSON API for the SIH 2026 problem: *where and at which
lead time is the NWP forecast likely to bust, and why?*

## Run (2 commands)

```bash
npm install
npm start
```

Open **http://localhost:3000** (dashboard). Needs Node 18+ and internet on
first load (React / Tailwind / Babel come from CDN).

## What you see

- **Geographic confidence map** — 36 subdivisions at real lat/lon, coloured by
  bust probability. Pick **Date** and **Lead day (1–10)**. Second tab shows a
  **real fine-grid snapshot** (real ECMWF vs real IMD, no downscaling, 5 demo dates).
- **Selected region panel** — forecast, observed, bust probability,
  confidence (= 1 − P_bust), and **major uncertainty drivers** (SHAP-based,
  real ERA5 variables).
- **Lead-time comparison** — bust probability Day 1→10 for the clicked region.
- **High-risk areas** and **error-prone regions** (whole test period).
- **Model metrics** — AUROC / AUPRC / Brier / LogLoss for each feature set.
- **Review queue** — click a row to jump the map to that date/lead/region.

## API (Node/Express, `server/server.js`)

| Endpoint | Purpose |
|---|---|
| `GET /api/meta` | dates, subdivisions, lead days, centroids |
| `GET /api/confidence-map?date=YYYY-MM-DD[&lead_day=N]` | region-wise confidence + bust probability |
| `GET /api/bust-probability?subdivision=&date=&lead_day=` | single cell |
| `GET /api/review-queue?limit=20[&subdivision=]` | ranked high-risk forecasts + reasons |
| `GET /api/explain?date=&subdivision=&lead_day=` | explanation for one cell |
| `GET /api/error-prone-areas?top_n=10` | regions ranked by mean bust probability |
| `GET /api/grid-snapshot?date=&lead_day=` | real fine-grid snapshot |
| `GET /api/model-info` | metrics, features, caveats |

Data in `server/data/*.json` is the exported output of the Python pipeline
(real IMD + real ECMWF + real ERA5, monsoon 2023–2024; trained on 2023,
tested on Jul–Sep 2024).

## Honest status

- Best AUROC ≈ 0.62 vs 0.606 forecast-amount baseline: modest, real gain from ERA5.
- Only 2 real forecast years; forecast downscaled from India-mean to subdivisions.
- No real ensemble (ENS) yet; historical-error features were tried and excluded (they hurt).
- No automated test suite yet.
- Subdivisions are approximate lat/lon boxes, not official IMD polygons.

## Layout

```
server/server.js       Express API + static hosting
server/data/           exported real-data model outputs (JSON)
public/index.html      Tailwind + React (CDN) shell
public/app.jsx         React dashboard
```

## Live pipeline (forecasting the future)

`live/` turns the trained model into a forward-looking tool: today's ECMWF forecast in, Day 1–10 bust probability per
subdivision out. Inputs are only what exists at forecast issue time (00Z run) — no observed rainfall.

```bash
pip install -r live/requirements.txt              # same versions the model was pickled with
python live/live_pipeline.py --node-data server/data   # latest ECMWF Open Data run  -> dashboard "Live forecast" tab
python live/live_pipeline.py --replay --node-data server/data   # offline demo from stored real inputs (labelled REPLAY)
python -m pytest live/tests -q                    # 5 tests
```
Run it on your machine/Colab (needs internet to ECMWF Open Data; `--source aws` if their server is slow).
Then refresh the dashboard: it re-reads `live_predictions.json` on every request, no restart needed.

What it does: latest 00Z HRES run → accumulated `tp` (steps 24–240 h) → India mean → de-accumulate → downscale to
subdivisions (real IMD climatology ratios) → step-0 fields (2t, 2d, msl, u/v/r @ 850/200 hPa) → dewpoint depression,
wind shear, RH850, 24 h MSLP tendency → XGBoost + isotonic → P(bust), confidence = 1 − P, top-3 drivers (TreeSHAP).

**Tested here:** feature logic, de-accumulation, downscaling, calibration, season/OOD flags, and *serve/train parity*
(stored real inputs through the live code reproduce the offline predictions, max diff < 0.001). GRIB-reading helpers were
checked on real ERA5 GRIB files. **Not tested here:** the actual download from ECMWF (no network route from the build
sandbox) — if a request fails, the error will name the parameter; the usual fix is `--source aws`.

Honest limitations of live mode
- **Season:** model trained on Jun–Sep only. Rows whose valid date is outside Jun–Sep are flagged `out_of_season` (unreliable).
- **Source mismatch:** trained on ERA5 state, live uses ECMWF step-0 analysis (same quantities, ~5-day ERA5 lag avoided).
- **Skill is modest** (AUROC ~0.62 vs 0.606 baseline) and calibrated probabilities are coarse (isotonic steps on few positives).
- **No verification yet:** a live forecast can only be scored after its dates pass and IMD observations arrive.
- Rows with features outside the training range are flagged per row (`ood_features`).
