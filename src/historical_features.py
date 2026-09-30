import numpy as np
import pandas as pd

HIST_FEATURE_COLS = ["hist_mae", "hist_bias", "hist_bust_freq", "roll30_mae"]


def add_historical_features(df: pd.DataFrame, train_cutoff_global_mean: dict = None):
    df = df.sort_values(["subdivision", "lead_day", "date"]).reset_index(drop=True)
    df["abs_err"] = (df["forecast_mm"] - df["observed_mm"]).abs()
    df["signed_err"] = df["forecast_mm"] - df["observed_mm"]

    g = df.groupby(["subdivision", "lead_day"])
    df["hist_mae"] = g["abs_err"].apply(lambda s: s.shift(1).expanding().mean()).reset_index(drop=True)
    df["hist_bias"] = g["signed_err"].apply(lambda s: s.shift(1).expanding().mean()).reset_index(drop=True)
    df["hist_bust_freq"] = g["bust"].apply(lambda s: s.shift(1).expanding().mean()).reset_index(drop=True) if "bust" in df else np.nan
    df["roll30_mae"] = g["abs_err"].apply(lambda s: s.shift(1).rolling(30, min_periods=3).mean()).reset_index(drop=True)

    if train_cutoff_global_mean is None:
        train_cutoff_global_mean = {c: df[c].mean() for c in HIST_FEATURE_COLS}
    for c in HIST_FEATURE_COLS:
        df[c] = df[c].fillna(train_cutoff_global_mean[c])
    return df, train_cutoff_global_mean
