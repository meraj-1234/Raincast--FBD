import json
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, log_loss
from xgboost import XGBClassifier

from labels import fit_p95_table, label_busts
from historical_features import add_historical_features

FEATURE_SETS = {
    "A_forecast_only":            ["forecast_mm", "lead_day", "month_sin", "month_cos"],
    "B_plus_climatology":         ["forecast_mm", "lead_day", "month_sin", "month_cos", "climatology_anom"],
    "C_plus_historical_error":    ["forecast_mm", "lead_day", "month_sin", "month_cos", "climatology_anom",
                                    "hist_mae", "hist_bias", "hist_bust_freq", "roll30_mae"],
    "C2_plus_lagged_spread_proxy": ["forecast_mm", "lead_day", "month_sin", "month_cos", "climatology_anom",
                                     "hist_mae", "hist_bias", "hist_bust_freq", "roll30_mae",
                                     "lagged_spread", "jumpiness"],
}


def metrics(y_true, p):
    return {
        "auroc": roc_auc_score(y_true, p),
        "auprc": average_precision_score(y_true, p),
        "brier": brier_score_loss(y_true, p),
        "log_loss": log_loss(y_true, p, labels=[0, 1]),
    }


def fit_and_score(model, X_train, y_train, X_calib, y_calib, X_test, y_test):
    model.fit(X_train, y_train)
    raw_calib = model.predict_proba(X_calib)[:, 1]
    iso = IsotonicRegression(out_of_bounds="clip").fit(raw_calib, y_calib)
    p_test = iso.transform(model.predict_proba(X_test)[:, 1])
    return metrics(y_test, p_test)


def main():
    df = pd.read_parquet("/manual/ps26079/data/dataset_real_forecast.parquet")
    df["date"] = pd.to_datetime(df["date"])

    train_mask = df["year"] == 2023
    p95 = fit_p95_table(df[train_mask])
    df = label_busts(df, p95)

    df_sorted = df.sort_values(["subdivision", "lead_day", "date"])
    train_only = df_sorted[df_sorted["year"] == 2023]
    _, frozen_means = add_historical_features(train_only.assign(bust=train_only["bust"]))
    df, _ = add_historical_features(df, train_cutoff_global_mean=frozen_means)

    train = df[df["year"] == 2023]
    calib = df[(df["year"] == 2024) & (df["date"] < "2024-06-21")]
    test = df[(df["year"] == 2024) & (df["date"] >= "2024-06-21")]

    print(f"train {len(train)} / calib {len(calib)} / test {len(test)}")
    print(f"bust prevalence -- train {train['bust'].mean():.3%}  test {test['bust'].mean():.3%}")
    print(f"bust counts -- train busts={train['bust'].sum()}  test busts={test['bust'].sum()}")
    print("\nbust rate by lead day (test):")
    print(test.groupby("lead_day")["bust"].mean().round(4))

    clim_rate = train.groupby(["subdivision", "month"])["bust"].mean().rename("p").reset_index()
    t = test.merge(clim_rate, on=["subdivision", "month"], how="left")
    p_clim = t["p"].fillna(train["bust"].mean()).values

    results = {"climatology": metrics(test["bust"].values, p_clim)}

    for name, cols in FEATURE_SETS.items():
        X_train, y_train = train[cols], train["bust"]
        X_calib, y_calib = calib[cols], calib["bust"]
        X_test, y_test = test[cols], test["bust"]
        pos, neg = y_train.sum(), (1 - y_train).sum()
        model = XGBClassifier(n_estimators=250, max_depth=4, learning_rate=0.05,
                               subsample=0.8, colsample_bytree=0.8,
                               scale_pos_weight=max(neg / max(pos, 1), 1.0),
                               eval_metric="auc", n_jobs=4)
        results[f"xgb_{name}"] = fit_and_score(model, X_train, y_train, X_calib, y_calib, X_test, y_test)

    best_ablation = max(
        (k for k in results if k.startswith("xgb_")),
        key=lambda k: results[k]["auroc"],
    )
    best_cols = FEATURE_SETS[best_ablation.replace("xgb_", "")]
    X_train, y_train = train[best_cols], train["bust"]
    X_calib, y_calib = calib[best_cols], calib["bust"]
    X_test, y_test = test[best_cols], test["bust"]

    models = {
        "logistic_regression": LogisticRegression(max_iter=2000, class_weight="balanced"),
        "random_forest": RandomForestClassifier(n_estimators=300, max_depth=6, class_weight="balanced", n_jobs=4, random_state=0),
    }
    for mname, model in models.items():
        results[f"model_{mname}"] = fit_and_score(model, X_train, y_train, X_calib, y_calib, X_test, y_test)
    results[f"model_xgboost"] = results[best_ablation]

    print("\n=== RESULTS ===")
    print(json.dumps(results, indent=2))
    print(f"\nBest feature set: {best_ablation} ({best_cols})")

    with open("/manual/ps26079/outputs/ablation_and_comparison.json", "w") as f:
        json.dump({
            "results": results,
            "best_feature_set": best_ablation,
            "best_feature_cols": best_cols,
            "dataset_summary": {
                "n_train": int(len(train)), "n_calib": int(len(calib)), "n_test": int(len(test)),
                "bust_count_train": int(train["bust"].sum()), "bust_count_test": int(test["bust"].sum()),
                "bust_rate_train": float(train["bust"].mean()), "bust_rate_test": float(test["bust"].mean()),
                "bust_rate_by_lead_day_test": test.groupby("lead_day")["bust"].mean().round(4).to_dict(),
                "n_subdivisions": int(df["subdivision"].nunique()),
                "train_period": "2023-06-01 to 2023-09-30", "test_period": "2024-07-21 to 2024-09-30",
            },
            "not_run_ablations": {
                "D_ens_spread": "no real ECMWF ensemble (ENS) data available in this environment",
                "E_meteorological": "no ERA5 access in this environment",
                "F_spatial_neighbourhood": "requires per-subdivision spatial grid over the full training "
                                            "period; only 5 demo dates fetched at grid resolution so far",
            },
        }, f, indent=2)
    print("saved outputs/ablation_and_comparison.json")


if __name__ == "__main__":
    main()
