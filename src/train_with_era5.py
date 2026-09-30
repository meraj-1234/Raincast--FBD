import json
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, log_loss
from xgboost import XGBClassifier

from labels import fit_p95_table, label_busts

FEATURE_SETS = {
    "A_forecast_only": ["forecast_mm", "lead_day", "month_sin", "month_cos"],
    "B_plus_climatology": ["forecast_mm", "lead_day", "month_sin", "month_cos", "climatology_anom"],
    "E_plus_era5_meteorological": ["forecast_mm", "lead_day", "month_sin", "month_cos", "climatology_anom",
                                    "dewpoint_depression", "wind_shear_850_200", "rh850", "msl_tendency"],
}


def metrics(y_true, p):
    return {"auroc": roc_auc_score(y_true, p), "auprc": average_precision_score(y_true, p),
            "brier": brier_score_loss(y_true, p), "log_loss": log_loss(y_true, p, labels=[0, 1])}


def fit_and_score(model, Xtr, ytr, Xc, yc, Xte, yte):
    model.fit(Xtr, ytr)
    raw_c = model.predict_proba(Xc)[:, 1]
    iso = IsotonicRegression(out_of_bounds="clip").fit(raw_c, yc)
    p = iso.transform(model.predict_proba(Xte)[:, 1])
    return metrics(yte, p), model, iso


def main():
    df = pd.read_parquet("/manual/ps26079/data/dataset_with_era5.parquet")
    df["msl_tendency"] = df["msl_tendency"].fillna(0)

    train_mask = df["year"] == 2023
    p95 = fit_p95_table(df[train_mask])
    df = label_busts(df, p95)

    train = df[df["year"] == 2023]
    calib = df[(df["year"] == 2024) & (df["date"] < "2024-06-21")]
    test = df[(df["year"] == 2024) & (df["date"] >= "2024-06-21")]
    print(f"train {len(train)} calib {len(calib)} test {len(test)}")

    clim_rate = train.groupby(["subdivision", "month"])["bust"].mean().rename("p").reset_index()
    t = test.merge(clim_rate, on=["subdivision", "month"], how="left")
    p_clim = t["p"].fillna(train["bust"].mean()).values
    results = {"climatology": metrics(test["bust"].values, p_clim)}

    best_model = None
    best_iso = None
    best_cols = None
    best_auroc = -1
    for name, cols in FEATURE_SETS.items():
        Xtr, ytr = train[cols], train["bust"]
        Xc, yc = calib[cols], calib["bust"]
        Xte, yte = test[cols], test["bust"]
        pos, neg = ytr.sum(), (1 - ytr).sum()
        model = XGBClassifier(n_estimators=250, max_depth=4, learning_rate=0.05,
                               subsample=0.8, colsample_bytree=0.8,
                               scale_pos_weight=max(neg / max(pos, 1), 1.0), eval_metric="auc", n_jobs=4)
        m, fitted, iso = fit_and_score(model, Xtr, ytr, Xc, yc, Xte, yte)
        results[f"xgb_{name}"] = m
        if m["auroc"] > best_auroc:
            best_auroc = m["auroc"]; best_model = fitted; best_iso = iso; best_cols = cols; best_name = name

    Xtr, ytr = train[best_cols], train["bust"]
    Xc, yc = calib[best_cols], calib["bust"]
    Xte, yte = test[best_cols], test["bust"]
    for mname, model in [("logistic_regression", LogisticRegression(max_iter=2000, class_weight="balanced")),
                          ("random_forest", RandomForestClassifier(n_estimators=300, max_depth=6, class_weight="balanced", n_jobs=4, random_state=0))]:
        m, _, _ = fit_and_score(model, Xtr, ytr, Xc, yc, Xte, yte)
        results[f"model_{mname}_bestfeat"] = m

    print(json.dumps(results, indent=2))
    print(f"\nBest feature set: {best_name}, cols={best_cols}")

    import pickle
    with open("/manual/ps26079/models/model_era5.pkl", "wb") as f:
        pickle.dump({"model": best_model, "calibrator": best_iso, "features": best_cols}, f)
    with open("/manual/ps26079/outputs/era5_results.json", "w") as f:
        json.dump({"results": results, "best_feature_set": best_name, "best_cols": best_cols}, f, indent=2)

    p_final = best_iso.transform(best_model.predict_proba(test[best_cols])[:, 1])
    test = test.copy(); test["bust_probability"] = p_final
    test.to_parquet("/manual/ps26079/outputs/scored_test_era5.parquet", index=False)
    print("saved model_era5.pkl, era5_results.json, scored_test_era5.parquet")


if __name__ == "__main__":
    main()
