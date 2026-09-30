import numpy as np
import pandas as pd

FEATURE_COLS = [
    "forecast_mm", "ens_spread", "jumpiness", "moisture_anom", "shear_anom",
    "lead_day", "month_sin", "month_cos", "climatology_anom", "spread_to_climo_ratio",
]


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["climatology_anom"] = df["forecast_mm"] - df["climatology_mm"]
    df["spread_to_climo_ratio"] = df["ens_spread"] / (df["climatology_mm"] + 1.0)
    return df
