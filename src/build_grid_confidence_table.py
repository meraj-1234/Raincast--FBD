import numpy as np
import pandas as pd
import xarray as xr

IMD_FILES = {
    2023: "/mnt/user-data/uploads/RF25_ind2023_rfp25.nc",
    2024: "/mnt/user-data/uploads/RF25_ind2024_rfp25.nc",
}


def nearest_imd_coord(vals, imd_vals):
    imd_vals = np.asarray(imd_vals)
    idx = np.searchsorted(imd_vals, vals)
    idx = np.clip(idx, 1, len(imd_vals) - 1)
    left, right = imd_vals[idx - 1], imd_vals[idx]
    return np.where(np.abs(vals - left) <= np.abs(vals - right), left, right)


def main():
    fc = pd.read_csv("/mnt/user-data/uploads/grid_confidence_table_RAW.csv")
    fc["valid_date"] = pd.to_datetime(fc["valid_date"])
    fc["year"] = fc["valid_date"].dt.year

    out_rows = []
    for year, path in IMD_FILES.items():
        yr_fc = fc[fc["year"] == year].copy()
        if yr_fc.empty:
            continue
        ds = xr.open_dataset(path)
        imd_lats = ds["LATITUDE"].values
        imd_lons = ds["LONGITUDE"].values

        yr_fc["imd_lat"] = nearest_imd_coord(yr_fc["latitude"].values, imd_lats)
        yr_fc["imd_lon"] = nearest_imd_coord(yr_fc["longitude"].values, imd_lons)

        rain = ds["RAINFALL"]
        for date, day_grp in yr_fc.groupby(yr_fc["valid_date"].dt.date):
            try:
                day_slice = rain.sel(TIME=str(date))
            except KeyError:
                continue
            obs_da = day_slice.sel(
                LATITUDE=xr.DataArray(day_grp["imd_lat"].values, dims="points"),
                LONGITUDE=xr.DataArray(day_grp["imd_lon"].values, dims="points"),
                method="nearest",
            )
            day_grp = day_grp.copy()
            day_grp["observed_mm"] = obs_da.values
            out_rows.append(day_grp)

    result = pd.concat(out_rows, ignore_index=True)
    result["observed_mm"] = result["observed_mm"].fillna(0).clip(lower=0)
    result["forecast_error_mm"] = result["forecast_mm"] - result["observed_mm"]

    final = result.rename(columns={
        "valid_date": "Date", "imd_lat": "Latitude", "imd_lon": "Longitude",
        "lead_day": "Lead Day", "forecast_mm": "Forecast Rainfall",
        "observed_mm": "IMD Observed Rainfall", "forecast_error_mm": "Forecast Error",
    })[["Date", "Latitude", "Longitude", "Lead Day", "Forecast Rainfall",
        "IMD Observed Rainfall", "Forecast Error"]]

    final.to_csv("/manual/ps26079/outputs/grid_confidence_table_FINAL.csv", index=False)
    print(f"saved {len(final):,} rows")
    print(final.describe())
    print(final.head())


if __name__ == "__main__":
    main()
