import json
import pickle
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

ASSETS_DIR = Path(__file__).parent / "assets"
LEADS = list(range(1, 11))
MET_FEATURES = ["dewpoint_depression", "wind_shear_850_200", "rh850", "msl_tendency"]

REASON_TEMPLATES = {
    "forecast_mm": "forecast amount ({v:.0f} mm/day) is unusually {dir} for this subdivision",
    "lead_day": "Day {v:.0f} forecast (uncertainty grows with lead time)",
    "month_sin": "time of season",
    "month_cos": "time of season",
    "climatology_anom": "forecast is {v:+.0f} mm/day away from the seasonal normal",
    "dewpoint_depression": "dewpoint depression is unusually {dir} ({v:.1f} K) -- {moist}",
    "wind_shear_850_200": "850-200 hPa wind shear is unusually {dir} ({v:.1f} m/s)",
    "rh850": "850 hPa relative humidity is unusually {dir} ({v:.0f}%)",
    "msl_tendency": "sea-level pressure changing unusually fast ({v:+.0f} Pa/24h) -- rapid {tend}",
}


def load_bundle():
    assets = json.loads((ASSETS_DIR / "assets.json").read_text())
    with open(ASSETS_DIR / "model_era5.pkl", "rb") as f:
        m = pickle.load(f)
    return {"assets": assets, "model": m["model"], "calibrator": m["calibrator"], "features": m["features"]}


def deaccumulate(cum: dict) -> dict:
    daily, prev = {}, 0.0
    for lead in sorted(cum):
        daily[lead] = max(0.0, float(cum[lead]) - prev)
        prev = float(cum[lead])
    return daily


def build_features(init_date: date, india_daily: dict, met: dict, assets: dict) -> pd.DataFrame:
    rows = []
    for sub in assets["boxes"]:
        for lead in LEADS:
            valid = init_date + timedelta(days=lead)
            m = valid.month
            fc = india_daily[lead] * assets["ratio"][sub][str(m)]
            row = {
                "subdivision": sub, "lead_day": lead, "valid_date": valid.isoformat(),
                "forecast_mm": fc,
                "month_sin": float(np.sin(2 * np.pi * m / 12)),
                "month_cos": float(np.cos(2 * np.pi * m / 12)),
                "climatology_anom": fc - assets["sub_climo"][sub][str(m)],
                "valid_month": m,
            }
            row.update({k: float(met[sub][k]) for k in MET_FEATURES})
            rows.append(row)
    return pd.DataFrame(rows)


def _reason(feat, val, contrib, median):
    tmpl = REASON_TEMPLATES.get(feat)
    if tmpl is None or feat in ("month_sin", "month_cos"):
        return None
    d = "high" if val >= median else "low"
    return tmpl.format(
        v=val, dir=d,
        moist="very moist column" if val < 2 else "drier column",
        tend="drop (deepening system)" if val < 0 else "rise",
    )


def predict(df: pd.DataFrame, bundle: dict, top_k: int = 3) -> pd.DataFrame:
    feats, assets = bundle["features"], bundle["assets"]
    X = df[feats]
    p_raw = bundle["model"].predict_proba(X)[:, 1]
    p = bundle["calibrator"].transform(p_raw)

    contribs = bundle["model"].get_booster().predict(xgb.DMatrix(X, feature_names=feats), pred_contribs=True)[:, :-1]
    stats = assets["feature_stats"]
    reasons, ood = [], []
    for i in range(len(df)):
        order = np.argsort(-np.abs(contribs[i]))
        rs = []
        for j in order:
            f = feats[j]
            r = _reason(f, float(X.iloc[i, j]), contribs[i, j], stats[f]["median"])
            if r:
                rs.append(r)
            if len(rs) == top_k:
                break
        reasons.append(rs)
        ood.append([f for f in feats if f not in ("lead_day", "month_sin", "month_cos")
                    and not (stats[f]["p005"] <= float(X.iloc[i][f]) <= stats[f]["p995"])])

    out = df.copy()
    out["bust_probability"] = p
    out["confidence"] = 1.0 - p
    out["reasons"] = reasons
    out["out_of_season"] = ~out["valid_month"].isin(assets["training_months"])
    out["ood_features"] = ood
    return out


def to_payload(pred: pd.DataFrame, init_date: date, mode: str, india_daily: dict, generated_at: str, extra_warnings=None):
    warnings = list(extra_warnings or [])
    n_off = int(pred["out_of_season"].sum())
    if n_off:
        warnings.append(
            f"{n_off} of {len(pred)} rows are for valid dates outside Jun-Sep. The model was trained only on "
            "monsoon (Jun-Sep 2023) -- treat those probabilities as unreliable.")
    n_ood = int((pred["ood_features"].str.len() > 0).sum())
    if n_ood:
        warnings.append(f"{n_ood} rows have input features outside the training range (flagged per row).")
    warnings.append("Model skill is modest (held-out AUROC ~0.62, baseline 0.606): use as a risk indicator, "
                    "not a precise probability. No verification exists yet for these dates.")
    recs = []
    for r in pred.itertuples():
        recs.append({
            "subdivision": r.subdivision, "lead_day": int(r.lead_day), "valid_date": r.valid_date,
            "forecast_mm": round(float(r.forecast_mm), 2),
            "bust_probability": round(float(r.bust_probability), 4),
            "confidence": round(float(r.confidence), 4),
            "reasons": r.reasons, "out_of_season": bool(r.out_of_season), "ood_features": r.ood_features,
        })
    return {
        "mode": mode, "init_date": init_date.isoformat(), "generated_at": generated_at,
        "india_mean_forecast_daily_mm": {str(k): round(v, 2) for k, v in india_daily.items()},
        "confidence_formula": "confidence = 1 - calibrated P(bust)",
        "warnings": warnings, "predictions": recs,
    }
