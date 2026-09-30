const express = require("express");
const cors = require("cors");
const path = require("path");
const fs = require("fs");

const app = express();
app.use(cors());
app.use(express.json());

const DATA = path.join(__dirname, "data");
const load = (f) => JSON.parse(fs.readFileSync(path.join(DATA, f), "utf-8"));

const confidence = load("confidence_data.json");
const reviewQueue = load("review_queue.json");
const errorProne = load("error_prone_areas.json");
const modelInfo = load("model_info.json");
const meta = load("meta.json");
const centroids = load("centroids.json");
const gridSnapshots = load("grid_snapshots.json");

const byDate = new Map();
for (const row of confidence) {
  if (!byDate.has(row.date)) byDate.set(row.date, []);
  byDate.get(row.date).push(row);
}

app.get("/api/meta", (_req, res) => res.json({ ...meta, centroids }));

app.get("/api/confidence-map", (req, res) => {
  const { date } = req.query;
  const lead = req.query.lead_day ? Number(req.query.lead_day) : null;
  if (!date || !byDate.has(date)) {
    return res.status(404).json({
      error: `No data for date ${date}`,
      available: { from: meta.dates[0], to: meta.dates[meta.dates.length - 1] },
    });
  }
  let rows = byDate.get(date);
  if (lead) rows = rows.filter((r) => r.lead_day === lead);
  res.json(
    rows.map((r) => ({
      subdivision: r.subdivision,
      lead_day: r.lead_day,
      bust_probability: r.bust_probability,
      confidence: Number((1 - r.bust_probability).toFixed(4)),
      forecast_mm: r.forecast_mm,
      observed_mm: r.observed_mm,
      actually_busted: r.bust === 1,
      lat: centroids[r.subdivision]?.[0],
      lon: centroids[r.subdivision]?.[1],
    }))
  );
});

app.get("/api/bust-probability", (req, res) => {
  const { subdivision, date } = req.query;
  const lead = Number(req.query.lead_day);
  const row = (byDate.get(date) || []).find(
    (r) => r.subdivision === subdivision && r.lead_day === lead
  );
  if (!row) return res.status(404).json({ error: "No matching row" });
  res.json({ ...row, confidence: Number((1 - row.bust_probability).toFixed(4)) });
});

app.get("/api/review-queue", (req, res) => {
  const limit = Number(req.query.limit) || 20;
  const { subdivision } = req.query;
  let rows = reviewQueue;
  if (subdivision) rows = rows.filter((r) => r.subdivision === subdivision);
  res.json(rows.slice(0, limit));
});

app.get("/api/explain", (req, res) => {
  const { date, subdivision } = req.query;
  const lead = Number(req.query.lead_day);
  const hit = reviewQueue.find(
    (r) => r.date === date && r.subdivision === subdivision && r.lead_day === lead
  );
  if (!hit) {
    return res.json({
      available: false,
      note: "Below the top-200 review-queue cutoff; no SHAP breakdown was computed for this cell.",
    });
  }
  res.json({ available: true, ...hit });
});

app.get("/api/error-prone-areas", (req, res) => {
  const n = Number(req.query.top_n) || 10;
  res.json(errorProne.slice(0, n));
});

app.get("/api/grid-snapshot", (req, res) => {
  const { date } = req.query;
  const lead = Number(req.query.lead_day);
  const snap = gridSnapshots.find((s) => s.date === date && s.lead_day === lead);
  if (!snap) return res.status(404).json({ error: "No snapshot" });
  res.json(snap);
});
app.get("/api/grid-snapshot-index", (_req, res) =>
  res.json(gridSnapshots.map((s) => ({ date: s.date, lead_day: s.lead_day })))
);

app.get("/api/live", (_req, res) => {
  const f = path.join(DATA, "live_predictions.json");
  if (!fs.existsSync(f)) {
    return res.status(404).json({
      error: "No live run yet",
      hint: "Run: python live/live_pipeline.py --node-data server/data   (or --replay for an offline demo)",
    });
  }
  res.json(JSON.parse(fs.readFileSync(f, "utf-8")));
});

app.get("/api/model-info", (_req, res) => {
  res.json({
    results: modelInfo.results,
    best_feature_set: modelInfo.best_feature_set,
    best_cols: modelInfo.best_cols,
    caveats: [
      "Only 2 real forecast years (2023 train, 2024 test); ECMWF Open Data does not go back to 2022.",
      "Forecast is India-domain-mean downscaled to subdivisions via real IMD climatology ratios.",
      "No real ensemble (ENS) data -- lagged-forecast spread is only a proxy and is excluded from the final model.",
      "ERA5 (humidity, wind shear, pressure tendency) is real and drives the gain over the forecast-amount baseline (0.606 -> ~0.62 AUROC).",
    ],
  });
});

const PUBLIC = path.join(__dirname, "..", "public");
app.use(express.static(PUBLIC));

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
  console.log(`PS26079 running -> http://localhost:${PORT}`);
  console.log(`API example     -> http://localhost:${PORT}/api/model-info`);
});
