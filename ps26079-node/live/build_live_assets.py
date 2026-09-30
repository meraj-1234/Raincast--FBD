import json, sys, shutil
from pathlib import Path
import numpy as np, pandas as pd, xarray as xr

ROOT = Path("/manual/ps26079")
sys.path.insert(0, str(ROOT / "src"))
from ingest_real_imd import SUBDIVISION_BOXES

OUT = ROOT / "live" / "assets"
FEATURES = ['forecast_mm', 'lead_day', 'month_sin', 'month_cos', 'climatology_anom',
            'dewpoint_depression', 'wind_shear_850_200', 'rh850', 'msl_tendency']

obs = pd.read_parquet(ROOT / "data/real_observed.parquet")
sub_climo = obs.groupby(["subdivision", "month"])["observed_mm"].mean()

frames = []
for y in (2023, 2024):
    ds = xr.open_dataset(f"/mnt/user-data/uploads/RF25_ind{y}_rfp25.nc")
    d = ds["RAINFALL"].mean(dim=["LATITUDE", "LONGITUDE"], skipna=True).to_dataframe(name="nat").reset_index()
    d["month"] = pd.to_datetime(d["TIME"]).dt.month
    frames.append(d)
national = pd.concat(frames).groupby("month")["nat"].mean()

assets = {
    "boxes": {k: list(v) for k, v in SUBDIVISION_BOXES.items()},
    "sub_climo": {s: {int(m): float(sub_climo[(s, m)]) for m in range(1, 13)} for s in SUBDIVISION_BOXES},
    "national_climo": {int(m): float(v) for m, v in national.items()},
    "features": FEATURES,
    "training_months": [6, 7, 8, 9],
}
assets["ratio"] = {s: {m: float(np.clip(assets["sub_climo"][s][m] / assets["national_climo"][m], 0.1, 5.0))
                       for m in range(1, 13)} for s in SUBDIVISION_BOXES}

df = pd.read_parquet(ROOT / "data/dataset_with_era5.parquet")
df["msl_tendency"] = df["msl_tendency"].fillna(0)
train = df[df["year"] == 2023]
assets["feature_stats"] = {f: {"p005": float(train[f].quantile(0.005)), "median": float(train[f].median()),
                               "p995": float(train[f].quantile(0.995))} for f in FEATURES}

(OUT / "assets.json").write_text(json.dumps(assets, indent=1))
shutil.copy(ROOT / "models/model_era5.pkl", OUT / "model_era5.pkl")
print("wrote", OUT / "assets.json", "and model_era5.pkl")
print({m: round(assets['national_climo'][m], 2) for m in (6, 7, 8, 9, 10)})
