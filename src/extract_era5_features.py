import numpy as np
import pandas as pd
import xarray as xr

from ingest_real_imd import SUBDIVISION_BOXES

SURFACE_GRIB = "/manual/ps26079/data/era5/era5_surface_2023_2024.grib"
PRESSURE_GRIB = "/manual/ps26079/data/era5/era5_pressure_2023_2024.grib"


def aggregate_by_subdivision(da: xr.DataArray, lat_name="latitude", lon_name="longitude") -> pd.DataFrame:
    rows = []
    for sub, (lat_min, lat_max, lon_min, lon_max) in SUBDIVISION_BOXES.items():
        box = da.sel({lat_name: slice(lat_max, lat_min), lon_name: slice(lon_min, lon_max)})
        daily = box.mean(dim=[lat_name, lon_name], skipna=True)
        df = daily.to_dataframe(name=da.name).reset_index()
        df["subdivision"] = sub
        rows.append(df)
    return pd.concat(rows, ignore_index=True)


def main():
    surf = xr.open_dataset(SURFACE_GRIB, engine="cfgrib", indexpath="")
    pres = xr.open_dataset(PRESSURE_GRIB, engine="cfgrib", indexpath="")

    dewpoint_depression = (surf["t2m"] - surf["d2m"]).rename("dewpoint_depression")
    msl = surf["msl"].rename("msl")
    wind10_speed = np.sqrt(surf["u10"] ** 2 + surf["v10"] ** 2).rename("wind10_speed")

    dd_df = aggregate_by_subdivision(dewpoint_depression)
    msl_df = aggregate_by_subdivision(msl)
    w10_df = aggregate_by_subdivision(wind10_speed)

    u850 = pres["u"].sel(isobaricInhPa=850)
    v850 = pres["v"].sel(isobaricInhPa=850)
    u200 = pres["u"].sel(isobaricInhPa=200)
    v200 = pres["v"].sel(isobaricInhPa=200)
    shear = np.sqrt((u850 - u200) ** 2 + (v850 - v200) ** 2).rename("wind_shear_850_200")
    rh850 = pres["r"].sel(isobaricInhPa=850).rename("rh850")

    shear_df = aggregate_by_subdivision(shear)
    rh850_df = aggregate_by_subdivision(rh850)

    merged = dd_df[["time", "subdivision", "dewpoint_depression"]]
    for extra in [msl_df[["time", "subdivision", "msl"]],
                  w10_df[["time", "subdivision", "wind10_speed"]],
                  shear_df[["time", "subdivision", "wind_shear_850_200"]],
                  rh850_df[["time", "subdivision", "rh850"]]]:
        merged = merged.merge(extra, on=["time", "subdivision"], how="left")

    merged = merged.rename(columns={"time": "init_date"})
    merged["init_date"] = pd.to_datetime(merged["init_date"]).dt.normalize()

    merged = merged.sort_values(["subdivision", "init_date"])
    merged["msl_tendency"] = merged.groupby("subdivision")["msl"].diff()

    merged.to_parquet("/manual/ps26079/data/era5_features.parquet", index=False)
    print(f"saved {len(merged):,} subdivision-days of real ERA5 features")
    print(merged.describe())


if __name__ == "__main__":
    main()
