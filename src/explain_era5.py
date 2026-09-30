import pickle
import numpy as np
import pandas as pd
import shap

import pickle as _pkl
with open("/manual/ps26079/models/model_era5.pkl","rb") as _f:
    FEATURE_COLS = _pkl.load(_f)["features"]

REASON_TEMPLATES = {
    "forecast_mm": "forecast amount ({v:.0f} mm/day) is unusually {dir} for what verifies at this lead time",
    "lead_day": "Day {v:.0f} forecast",
    "climatology_anom": "forecast is {v:+.0f} mm/day away from the seasonal normal for this subdivision",
    "dewpoint_depression": "dewpoint depression is unusually {dir} ({v:.1f}\u00b0C) -- {lowhigh_moisture}",
    "wind_shear_850_200": "850-200hPa wind shear is unusually {dir} ({v:.1f} m/s)",
    "rh850": "850hPa relative humidity is unusually {dir} ({v:.0f}%)",
    "msl_tendency": "sea-level pressure is changing unusually fast ({v:+.1f} Pa/24h) -- rapid pressure {tendency_dir}",
}


def _extra_fill(col, val):
    if col == "dewpoint_depression":
        return {"lowhigh_moisture": "very moist column" if val < 2 else "drier column"}
    if col == "msl_tendency":
        return {"tendency_dir": "drop (deepening system)" if val < 0 else "rise"}
    return {}


def reason_for_row(row: pd.Series, shap_row: np.ndarray, medians: pd.Series, top_k=3) -> list:
    order = np.argsort(-np.abs(shap_row))[:top_k]
    reasons = []
    for idx in order:
        col = FEATURE_COLS[idx]
        val = row[col]
        direction = "high" if val >= medians[col] else "low"
        tmpl = REASON_TEMPLATES.get(col)
        if tmpl is None:
            continue
        try:
            reasons.append(tmpl.format(v=val, dir=direction, **_extra_fill(col, val)))
        except Exception:
            continue
    return reasons


def main():
    with open("/manual/ps26079/models/model_era5.pkl", "rb") as f:
        bundle = pickle.load(f)
    model = bundle["model"]

    scored = pd.read_parquet("/manual/ps26079/outputs/scored_test_era5.parquet")
    top = scored.sort_values("bust_probability", ascending=False).head(200).reset_index(drop=True)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(top[FEATURE_COLS])
    medians = scored[FEATURE_COLS].median()

    records = []
    for i, row in top.iterrows():
        reasons = reason_for_row(row, shap_values[i], medians)
        records.append({
            "date": str(row["date"].date()), "subdivision": row["subdivision"],
            "lead_day": int(row["lead_day"]), "bust_probability": float(row["bust_probability"]),
            "actually_busted": int(row["bust"]), "forecast_mm": float(row["forecast_mm"]),
            "observed_mm": float(row["observed_mm"]), "reasons": reasons,
        })

    import json
    with open("/manual/ps26079/outputs/review_queue_era5.json", "w") as f:
        json.dump(records, f, indent=2)
    print(f"wrote {len(records)} explained review-queue rows")
    print(json.dumps(records[0], indent=2))


if __name__ == "__main__":
    main()
