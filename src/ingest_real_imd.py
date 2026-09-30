import numpy as np
import pandas as pd
import xarray as xr

SUBDIVISION_BOXES = {
    "Andaman & Nicobar Islands":        (6.0, 14.0, 92.0, 94.0),
    "Arunachal Pradesh":                (26.5, 29.5, 91.5, 97.5),
    "Assam & Meghalaya":                (24.0, 28.0, 89.5, 96.0),
    "Nagaland Manipur Mizoram Tripura": (21.5, 27.0, 91.5, 96.0),
    "Sub Himalayan West Bengal & Sikkim": (26.0, 28.0, 87.5, 89.5),
    "Gangetic West Bengal":             (21.5, 24.5, 85.5, 89.5),
    "Odisha":                           (17.5, 22.5, 81.5, 87.5),
    "Jharkhand":                        (21.5, 25.5, 83.0, 88.0),
    "Bihar":                            (24.0, 27.5, 83.5, 88.5),
    "East Uttar Pradesh":               (24.5, 28.5, 79.5, 84.5),
    "West Uttar Pradesh":               (26.5, 30.5, 77.0, 81.5),
    "Uttarakhand":                      (28.5, 31.5, 77.5, 81.0),
    "Haryana Chandigarh Delhi":         (27.5, 31.0, 74.5, 77.5),
    "Punjab":                           (29.5, 32.5, 73.5, 76.5),
    "Himachal Pradesh":                 (30.5, 33.5, 75.5, 79.0),
    "Jammu & Kashmir":                  (32.0, 37.5, 73.0, 80.5),
    "West Rajasthan":                   (24.0, 30.0, 69.0, 74.0),
    "East Rajasthan":                   (24.0, 28.0, 74.0, 78.0),
    "West Madhya Pradesh":              (21.0, 25.0, 74.5, 78.5),
    "East Madhya Pradesh":              (21.0, 25.0, 78.5, 82.5),
    "Gujarat Region":                   (20.0, 24.5, 68.5, 74.5),
    "Saurashtra & Kutch":               (20.5, 24.0, 68.5, 72.0),
    "Konkan & Goa":                     (15.0, 20.0, 72.5, 74.0),
    "Madhya Maharashtra":               (17.0, 21.0, 73.5, 76.5),
    "Marathwada":                       (17.5, 20.5, 74.5, 78.0),
    "Vidarbha":                         (19.0, 21.5, 77.0, 80.5),
    "Chhattisgarh":                     (17.5, 24.0, 80.0, 84.5),
    "Coastal Andhra Pradesh":           (13.0, 20.0, 79.0, 84.5),
    "Telangana":                        (16.0, 19.5, 77.0, 81.5),
    "Rayalaseema":                      (12.5, 16.0, 76.5, 80.0),
    "Tamil Nadu Puducherry":            (8.0, 13.5, 76.5, 80.5),
    "Coastal Karnataka":                (12.0, 15.0, 74.0, 75.5),
    "North Interior Karnataka":         (14.5, 18.0, 74.5, 78.0),
    "South Interior Karnataka":         (11.5, 15.0, 75.0, 78.5),
    "Kerala":                           (8.0, 12.8, 74.5, 77.5),
    "Lakshadweep":                      (8.0, 12.5, 71.5, 74.0),
}

FILES = {
    2022: "/mnt/user-data/uploads/RF25_ind2022_rfp25.nc",
    2023: "/mnt/user-data/uploads/RF25_ind2023_rfp25.nc",
    2024: "/mnt/user-data/uploads/RF25_ind2024_rfp25.nc",
    2025: "/mnt/user-data/uploads/RF25_ind2025_rfp25.nc",
}


def aggregate_year(path: str, year: int) -> pd.DataFrame:
    ds = xr.open_dataset(path)
    rows = []
    for sub, (lat_min, lat_max, lon_min, lon_max) in SUBDIVISION_BOXES.items():
        box = ds["RAINFALL"].sel(
            LATITUDE=slice(lat_min, lat_max), LONGITUDE=slice(lon_min, lon_max)
        )
        daily_mean = box.mean(dim=["LATITUDE", "LONGITUDE"], skipna=True)
        df = daily_mean.to_dataframe(name="observed_mm").reset_index()
        df["subdivision"] = sub
        df["year"] = year
        rows.append(df[["TIME", "subdivision", "year", "observed_mm"]])
    out = pd.concat(rows, ignore_index=True).rename(columns={"TIME": "date"})
    return out


def main():
    all_years = []
    for year, path in FILES.items():
        df = aggregate_year(path, year)
        n_nan = df["observed_mm"].isna().sum()
        print(f"{year}: {len(df)} rows, {n_nan} NaN (box partly outside land/data domain)")
        all_years.append(df)
    real = pd.concat(all_years, ignore_index=True)
    real["observed_mm"] = real["observed_mm"].fillna(0.0).clip(lower=0)
    real["month"] = pd.to_datetime(real["date"]).dt.month
    real["doy"] = pd.to_datetime(real["date"]).dt.dayofyear

    print(real.groupby("subdivision")["observed_mm"].agg(["mean", "max"]).round(2))
    real.to_parquet("/manual/ps26079/data/real_observed.parquet", index=False)
    print(f"\nsaved {len(real):,} real subdivision-days -> data/real_observed.parquet")


if __name__ == "__main__":
    main()
