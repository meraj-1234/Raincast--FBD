import json
import pickle
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss
from xgboost import XGBClassifier

from labels import fit_p95_table, label_busts
from build_real_forecast_dataset import FEATURE_COLS_REAL


def decision_cost(y_true, p_pred, thresh=0.15, miss_cost=10.0, false_alarm_cost=1.0):
    flag = (p_pred >= thresh).astype(int)
    miss = ((flag == 0) & (y_true == 1)).sum() * miss_cost
    fa = ((flag == 1) & (y_true == 0)).sum() * false_alarm_cost
    return (miss + fa) / len(y_true) * 1000


def main():
    df = pd.read_parquet("/manual/ps26079/data/dataset_real_forecast.parquet")
    df["date"] = pd.to_datetime(df["date"])

    train = df[df["year"] == 2023]
    calib = df[(df["year"] == 2024) & (df["date"] < "2024-06-21")]
    test = df[(df["year"] == 2024) & (df["date"] >= "2024-06-21")]

    p95 = fit_p95_table(train)
    train = label_busts(train, p95)
    calib = label_busts(calib, p95)
    test = label_busts(test, p95)

    print(f"train {len(train)} / calib {len(calib)} / test {len(test)} rows")
    print(f"bust rate -- train {train['bust'].mean():.3%}  test {test['bust'].mean():.3%}")

    X_train, y_train = train[FEATURE_COLS_REAL], train["bust"]
    X_calib, y_calib = calib[FEATURE_COLS_REAL], calib["bust"]
    X_test, y_test = test[FEATURE_COLS_REAL], test["bust"]

    results = {}

    clim_rate = train.groupby(["subdivision", "month"])["bust"].mean().rename("p").reset_index()
    t = test.merge(clim_rate, on=["subdivision", "month"], how="left")
    p_clim = t["p"].fillna(train["bust"].mean()).values
    results["climatology"] = {"auroc": roc_auc_score(y_test, p_clim), "brier": brier_score_loss(y_test, p_clim)}

    lr_amt = LogisticRegression(max_iter=1000).fit(X_train[["forecast_mm"]], y_train)
    p_amt = lr_amt.predict_proba(X_test[["forecast_mm"]])[:, 1]
    results["forecast_amount_only"] = {"auroc": roc_auc_score(y_test, p_amt), "brier": brier_score_loss(y_test, p_amt)}

    lr_spread = LogisticRegression(max_iter=1000).fit(X_train[["lagged_spread", "lead_day"]], y_train)
    p_spread = lr_spread.predict_proba(X_test[["lagged_spread", "lead_day"]])[:, 1]
    results["lagged_spread_baseline"] = {"auroc": roc_auc_score(y_test, p_spread), "brier": brier_score_loss(y_test, p_spread)}

    pos, neg = y_train.sum(), (1 - y_train).sum()
    model = XGBClassifier(
        n_estimators=250, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=max(neg / max(pos, 1), 1.0), eval_metric="auc", n_jobs=4,
    )
    model.fit(X_train, y_train)

    raw_calib_p = model.predict_proba(X_calib)[:, 1]
    iso = IsotonicRegression(out_of_bounds="clip").fit(raw_calib_p, y_calib)
    p_model = iso.transform(model.predict_proba(X_test)[:, 1])

    results["xgboost_calibrated"] = {
        "auroc": roc_auc_score(y_test, p_model), "brier": brier_score_loss(y_test, p_model),
        "decision_cost_per_1000": decision_cost(y_test.values, p_model),
    }
    for k, pv in [("climatology", p_clim), ("forecast_amount_only", p_amt), ("lagged_spread_baseline", p_spread)]:
        results[k]["decision_cost_per_1000"] = decision_cost(y_test.values, pv)

    print(json.dumps(results, indent=2))

    with open("/manual/ps26079/models/model_realfc.pkl", "wb") as f:
        pickle.dump({"model": model, "calibrator": iso, "features": FEATURE_COLS_REAL}, f)

    with open("/manual/ps26079/outputs/eval_results_realfc.json", "w") as f:
        json.dump({
            "results": results, "n_test_rows": int(len(test)),
            "train_period": "2023-06-01 to 2023-09-30 (full monsoon)",
            "calib_period": "2024-06-01 to 2024-06-20",
            "test_period": "2024-06-21 to 2024-09-30 (held out)",
            "note": "Train/test split is across TIME within available years (2023-2024 only); "
                    "ECMWF Open Data's archive does not extend back to 2022, and 2025 forecast "
                    "was not fetched, so this is not an independent-year holdout like the earlier "
                    "hybrid run. forecast_mm is downscaled from ECMWF's India-domain-mean tp using "
                    "real IMD subdivision climatology ratios (see build_real_forecast_dataset.py).",
        }, f, indent=2)

    test = test.copy()
    test["bust_probability"] = p_model
    test.to_parquet("/manual/ps26079/outputs/scored_test_realfc.parquet", index=False)
    print("saved model_realfc.pkl, eval_results_realfc.json, scored_test_realfc.parquet")


if __name__ == "__main__":
    main()
