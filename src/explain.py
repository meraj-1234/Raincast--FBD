import pickle
import numpy as np
import pandas as pd
import shap

from features import FEATURE_COLS

REASON_TEMPLATES = {
    "forecast_mm": "forecast amount ({v:.0f} mm/day) is well above what usually verifies at this lead time",
    "ens_spread": "ensemble spread is unusually {dir} ({v:.1f} mm) for this situation",
    "jumpiness": "this forecast has been disagreeing sharply run-to-run ({v:.1f} mm jump)",
    "moisture_anom": "column moisture is {dir} normal ({v:+.2f} sd)",
    "shear_anom": "wind shear is {dir} normal ({v:+.2f} sd), consistent with a rapidly evolving system",
    "lead_day": "this is a Day {v:.0f} forecast, where errors are historically larger",
    "climatology_anom": "forecast is {v:+.0f} mm/day away from the seasonal normal for this subdivision",
    "spread_to_climo_ratio": "ensemble spread is {dir} relative to the season's typical rainfall",
}


def reason_for_row(row: pd.Series, shap_row: np.ndarray, top_k=3) -> list:
    order = np.argsort(-np.abs(shap_row))[:top_k]
    reasons = []
    for idx in order:
        col = FEATURE_COLS[idx]
        val = row[col]
        direction = "low" if shap_row[idx] * val < 0 and val < 0 else ("high" if val > 0 else "low")
        tmpl = REASON_TEMPLATES.get(col)
        if tmpl is None:
            continue
        try:
            reasons.append(tmpl.format(v=val, dir=direction))
        except Exception:
            continue
    return reasons


def main():
    with open("/manual/ps26079/models/model.pkl", "rb") as f:
        bundle = pickle.load(f)
    model = bundle["model"]

    scored = pd.read_parquet("/manual/ps26079/outputs/scored_test_2022.parquet")
    top = scored.sort_values("bust_probability", ascending=False).head(200).reset_index(drop=True)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(top[FEATURE_COLS])

    records = []
    for i, row in top.iterrows():
        reasons = reason_for_row(row, shap_values[i])
        records.append({
            "date": str(row["date"].date()), "subdivision": row["subdivision"],
            "lead_day": int(row["lead_day"]), "bust_probability": float(row["bust_probability"]),
            "actually_busted": int(row["bust"]), "forecast_mm": float(row["forecast_mm"]),
            "observed_mm": float(row["observed_mm"]), "reasons": reasons,
        })

    import json
    with open("/manual/ps26079/outputs/review_queue.json", "w") as f:
        json.dump(records, f, indent=2)
    print(f"wrote {len(records)} explained review-queue rows")
    print(json.dumps(records[0], indent=2))


if __name__ == "__main__":
    main()
