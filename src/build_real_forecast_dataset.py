import numpy as np
import pandas as pd

FEATURE_COLS_REAL = [
    "forecast_mm", "lagged_spread", "jumpiness", "lead_day",
    "month_sin", "month_cos", "climatology_anom",
]

MONSOON_MONTHS = [6, 7, 8, 9]


def load_national_climo():
    import xarray as xr
    frames = []
    for year, path in {
        2023: "/mnt/user-data/uploads/RF25_ind2023_rfp25.nc",
        2024: "/mnt/user-data/uploads/RF25_ind2024_rfp25.nc",
    }.items():
        ds = xr.open_dataset(path)
        daily = ds["RAINFALL"].mean(dim=["LATITUDE", "LONGITUDE"], skipna=True)
        df = daily.to_dataframe(name="national_mm").reset_index().rename(columns={"TIME": "date"})
        frames.append(df)
    nat = pd.concat(frames, ignore_index=True)
    nat["month"] = pd.to_datetime(nat["date"]).dt.month
    return nat.groupby("month")["national_mm"].mean()


def main():
    fc = pd.read_parquet("/manual/ps26079/data/ecmwf_forecast_combined.parquet")
    fc["init_date"] = pd.to_datetime(fc["init_date"])

    fc = fc.sort_values(["init_date", "lead_day"])
    fc["daily_tp_mm"] = fc.groupby("init_date")["india_mean_tp_mm"].diff()
    fc.loc[fc["lead_day"] == 1, "daily_tp_mm"] = fc.loc[fc["lead_day"] == 1, "india_mean_tp_mm"]
    fc["daily_tp_mm"] = fc["daily_tp_mm"].clip(lower=0)
    fc["india_mean_tp_mm"] = fc["daily_tp_mm"]
    fc = fc.drop(columns=["daily_tp_mm"])

    fc["valid_date"] = fc["init_date"] + pd.to_timedelta(fc["lead_hours"], unit="h")
    fc["valid_date"] = fc["valid_date"].dt.normalize()

    obs = pd.read_parquet("/manual/ps26079/data/real_observed.parquet")
    obs["date"] = pd.to_datetime(obs["date"])
    obs_monsoon = obs[obs["month"].isin(MONSOON_MONTHS)]
    sub_climo = obs_monsoon.groupby(["subdivision", "month"])["observed_mm"].mean().rename("sub_climo").reset_index()

    national_climo = load_national_climo()

    wide = fc.pivot_table(index="valid_date", columns="lead_day", values="india_mean_tp_mm")
    lagged_spread = wide.std(axis=1, skipna=True).rename("lagged_spread")
    jump = wide.diff(axis=1).abs().mean(axis=1, skipna=True).rename("jumpiness")

    fc = fc.merge(lagged_spread, on="valid_date", how="left")
    fc = fc.merge(jump, on="valid_date", how="left")
    fc["lagged_spread"] = fc["lagged_spread"].fillna(fc["lagged_spread"].median())
    fc["jumpiness"] = fc["jumpiness"].fillna(fc["jumpiness"].median())

    fc["month"] = fc["valid_date"].dt.month
    fc["year"] = fc["valid_date"].dt.year

    subs = sub_climo["subdivision"].unique()
    rows = []
    for sub in subs:
        s = fc.copy()
        s["subdivision"] = sub
        s = s.merge(sub_climo[sub_climo["subdivision"] == sub][["month", "sub_climo"]], on="month", how="left")
        s["national_climo"] = s["month"].map(national_climo)
        s["ratio"] = (s["sub_climo"] / s["national_climo"]).clip(0.1, 5.0)
        s["forecast_mm"] = (s["india_mean_tp_mm"] * s["ratio"]).clip(lower=0)
        rows.append(s)
    fc_sub = pd.concat(rows, ignore_index=True)

    df = fc_sub.merge(
        obs[["subdivision", "date", "observed_mm"]],
        left_on=["subdivision", "valid_date"], right_on=["subdivision", "date"], how="inner",
    )
    df["climatology_mm"] = df["sub_climo"]
    df["climatology_anom"] = df["forecast_mm"] - df["climatology_mm"]
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    keep = ["date", "year", "month", "subdivision", "lead_day", "forecast_mm", "observed_mm",
            "lagged_spread", "jumpiness", "climatology_mm", "climatology_anom",
            "month_sin", "month_cos"]
    df = df[keep].dropna()
    df.to_parquet("/manual/ps26079/data/dataset_real_forecast.parquet", index=False)
    print(f"built {len(df):,} rows -- REAL forecast (downscaled) x REAL observed")
    print(df.groupby("year").size())
    print(f"\nsubdivision downscaling ratios (sanity check, should be ~0.3-2.5):")
    print(fc_sub.groupby("subdivision")["ratio"].mean().round(2))


if __name__ == "__main__":
    main()
