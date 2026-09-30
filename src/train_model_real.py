import json
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss
from xgboost import XGBClassifier

from labels import fit_p95_table, label_busts
from features import add_features, FEATURE_COLS

HELD_OUT_YEAR = 2025
CALIB_YEARS = [2023, 2024]
TRAIN_YEARS = [2022, 2023, 2024]


def decision_cost(y_true, p_pred, thresh=0.15, miss_cost=10.0, false_alarm_cost=1.0):
    flag = (p_pred >= thresh).astype(int)
    miss = ((flag == 0) & (y_true == 1)).sum() * miss_cost
    fa = ((flag == 1) & (y_true == 0)).sum() * false_alarm_cost
    return (miss + fa) / len(y_true) * 1000


def main():
    df = pd.read_parquet("/manual/ps26079/data/dataset_hybrid.parquet")

    train_raw = df[df["year"].isin(TRAIN_YEARS)]
    p95 = fit_p95_table(train_raw)
    df = label_busts(df, p95)
    df = add_features(df)

    print(f"overall bust rate: {df['bust'].mean():.3%}")
    print(df.groupby("lead_day")["bust"].mean().round(4).to_string())

    train = df[df["year"].isin(TRAIN_YEARS)]
    calib = df[df["year"].isin(CALIB_YEARS)]
    test = df[df["year"] == HELD_OUT_YEAR]
    test_band = test[test["lead_day"].between(3, 7)]

    X_train, y_train = train[FEATURE_COLS], train["bust"]
    X_calib, y_calib = calib[FEATURE_COLS], calib["bust"]
    X_test, y_test = test_band[FEATURE_COLS], test_band["bust"]

    results = {}

    clim_rate = train.groupby(["subdivision", "month"])["bust"].mean().rename("p").reset_index()
    t = test_band.merge(clim_rate, on=["subdivision", "month"], how="left")
    p_clim = t["p"].fillna(train["bust"].mean()).values
    results["climatology"] = {"auroc": roc_auc_score(y_test, p_clim), "brier": brier_score_loss(y_test, p_clim)}

    lr_amt = LogisticRegression(max_iter=1000).fit(X_train[["forecast_mm"]], y_train)
    p_amt = lr_amt.predict_proba(X_test[["forecast_mm"]])[:, 1]
    results["forecast_amount_only"] = {"auroc": roc_auc_score(y_test, p_amt), "brier": brier_score_loss(y_test, p_amt)}

    lr_spread = LogisticRegression(max_iter=1000).fit(X_train[["ens_spread", "lead_day"]], y_train)
    p_spread = lr_spread.predict_proba(X_test[["ens_spread", "lead_day"]])[:, 1]
    results["ensemble_spread_baseline"] = {"auroc": roc_auc_score(y_test, p_spread), "brier": brier_score_loss(y_test, p_spread)}

    pos, neg = y_train.sum(), (1 - y_train).sum()
    model = XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=neg / pos, eval_metric="auc", n_jobs=4,
    )
    model.fit(X_train, y_train)

    raw_calib_p = model.predict_proba(X_calib)[:, 1]
    iso = IsotonicRegression(out_of_bounds="clip").fit(raw_calib_p, y_calib)

    raw_test_p = model.predict_proba(X_test)[:, 1]
    p_model = iso.transform(raw_test_p)
    results["xgboost_calibrated"] = {
        "auroc": roc_auc_score(y_test, p_model),
        "brier": brier_score_loss(y_test, p_model),
        "decision_cost_per_1000": decision_cost(y_test.values, p_model),
    }
    for k in ["climatology", "forecast_amount_only", "ensemble_spread_baseline"]:
        pv = {"climatology": p_clim, "forecast_amount_only": p_amt, "ensemble_spread_baseline": p_spread}[k]
        results[k]["decision_cost_per_1000"] = decision_cost(y_test.values, pv)

    print(json.dumps(results, indent=2))

    importances = dict(zip(FEATURE_COLS, model.feature_importances_.tolist()))

    import pickle
    with open("/manual/ps26079/models/model_real.pkl", "wb") as f:
        pickle.dump({"model": model, "calibrator": iso, "features": FEATURE_COLS}, f)

    with open("/manual/ps26079/outputs/eval_results_real.json", "w") as f:
        json.dump({"results": results, "feature_importance": importances,
                   "held_out_year": HELD_OUT_YEAR, "band": "lead_day 3-7",
                   "n_test_rows": int(len(test_band))}, f, indent=2)

    X_full, y_full = test[FEATURE_COLS], test["bust"]
    p_full = iso.transform(model.predict_proba(X_full)[:, 1])
    test = test.copy()
    test["bust_probability"] = p_full
    test.to_parquet("/manual/ps26079/outputs/scored_test_real.parquet", index=False)
    print("saved model_real.pkl, eval_results_real.json, scored_test_real.parquet")


if __name__ == "__main__":
    main()
