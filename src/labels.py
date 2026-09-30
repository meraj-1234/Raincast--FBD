import numpy as np
import pandas as pd

CATEGORY_EDGES = [2.5, 15.6, 64.5, 115.6, 204.5]


def rain_category(mm: np.ndarray) -> np.ndarray:
    return np.digitize(mm, CATEGORY_EDGES)


def fit_p95_table(train_df: pd.DataFrame) -> pd.DataFrame:
    train_df = train_df.copy()
    train_df["abs_err"] = (train_df["forecast_mm"] - train_df["observed_mm"]).abs()
    p95 = (
        train_df.groupby(["subdivision", "month"])["abs_err"]
        .quantile(0.95)
        .rename("p95_err")
        .reset_index()
    )
    return p95


def label_busts(df: pd.DataFrame, p95_table: pd.DataFrame) -> pd.DataFrame:
    df = df.merge(p95_table, on=["subdivision", "month"], how="left")
    df["p95_err"] = df["p95_err"].fillna(10.0)
    abs_err = (df["forecast_mm"] - df["observed_mm"]).abs()
    cat_f = rain_category(df["forecast_mm"].values)
    cat_o = rain_category(df["observed_mm"].values)
    magnitude_ok = abs_err >= np.maximum(10.0, df["p95_err"])
    flips = cat_f != cat_o
    significant = np.maximum(df["forecast_mm"], df["observed_mm"]) >= 15.6
    df["bust"] = (magnitude_ok & flips & significant).astype(int)
    df["abs_err"] = abs_err
    return df
