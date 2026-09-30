import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)

SUBDIVISIONS = [
    "Assam & Meghalaya", "Konkan & Goa", "Coastal Andhra Pradesh",
    "Coastal Karnataka", "East Rajasthan", "West Rajasthan",
    "Gangetic West Bengal", "Vidarbha", "Marathwada", "Kerala",
    "Odisha", "Saurashtra & Kutch",
]

BASE_CLIMO = {
    "Assam & Meghalaya": 9.5, "Konkan & Goa": 12.0, "Coastal Andhra Pradesh": 5.0,
    "Coastal Karnataka": 11.0, "East Rajasthan": 3.0, "West Rajasthan": 1.5,
    "Gangetic West Bengal": 8.0, "Vidarbha": 5.5, "Marathwada": 4.0,
    "Kerala": 10.5, "Odisha": 7.0, "Saurashtra & Kutch": 2.5,
}

YEARS = [2019, 2020, 2021, 2022]
LEADS = list(range(1, 11))


def seasonal_factor(doy: np.ndarray) -> np.ndarray:
    return np.clip(1.0 + 2.6 * np.exp(-((doy - 195) ** 2) / (2 * 55 ** 2)), 0.15, None)


def generate(n_years=YEARS) -> pd.DataFrame:
    rows = []
    for year in n_years:
        dates = pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="D")
        doy = dates.dayofyear.values
        season = seasonal_factor(doy)
        for sub in SUBDIVISIONS:
            climo = BASE_CLIMO[sub]
            shape = 0.55
            scale = climo * season / shape
            observed = RNG.gamma(shape, scale)

            regime_prob = 0.04 * (season / season.max())
            regime = RNG.random(len(dates)) < regime_prob

            for lead in LEADS:
                base_noise_sd = 0.10 + 0.045 * lead
                err_sd = np.where(regime, base_noise_sd * (2.2 + 0.15 * lead), base_noise_sd)
                mult_noise = RNG.normal(1.0, err_sd)
                bias = np.where(regime, RNG.normal(-0.35, 0.2, len(dates)), 0.0)
                forecast = np.clip(observed * (mult_noise + bias), 0, None)

                true_uncertainty = err_sd * (climo + 1)
                spread_collapse = np.where(regime, RNG.uniform(0.35, 0.65, len(dates)), 1.0)
                ens_spread = np.clip(true_uncertainty * spread_collapse * RNG.normal(1.0, 0.15, len(dates)), 0.3, None)

                jumpiness = np.abs(RNG.normal(0, np.where(regime, 6.0, 1.2), len(dates)))

                moisture_anom = np.where(regime, RNG.normal(1.1, 0.4, len(dates)), RNG.normal(0, 0.5, len(dates)))
                shear_anom = np.where(regime, RNG.normal(-0.8, 0.3, len(dates)), RNG.normal(0, 0.5, len(dates)))

                rows.append(pd.DataFrame({
                    "date": dates, "year": year, "month": dates.month, "doy": doy,
                    "subdivision": sub, "lead_day": lead,
                    "forecast_mm": forecast, "observed_mm": observed,
                    "ens_spread": ens_spread, "jumpiness": jumpiness,
                    "moisture_anom": moisture_anom, "shear_anom": shear_anom,
                    "climatology_mm": climo * season,
                    "regime_flag": regime.astype(int),
                }))
    return pd.concat(rows, ignore_index=True)


if __name__ == "__main__":
    df = generate()
    df.to_parquet("/manual/ps26079/data/dataset_raw.parquet", index=False)
    print(f"generated {len(df):,} rows across {len(SUBDIVISIONS)} subdivisions, "
          f"{len(YEARS)} years, {len(LEADS)} lead days")
    print(f"regime-day fraction: {df['regime_flag'].mean():.3%}")
