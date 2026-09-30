import numpy as np
import pandas as pd

RNG = np.random.default_rng(7)
LEADS = list(range(1, 11))


def build():
    obs = pd.read_parquet("/manual/ps26079/data/real_observed.parquet")
    obs = obs.sort_values(["subdivision", "date"]).reset_index(drop=True)

    climo_month = obs.groupby(["subdivision", "month"])["observed_mm"].mean().rename("climo_month").reset_index()
    obs = obs.merge(climo_month, on=["subdivision", "month"], how="left")

    rows = []
    for sub, g in obs.groupby("subdivision"):
        g = g.reset_index(drop=True)
        n = len(g)
        observed = g["observed_mm"].values
        climo = g["climo_month"].values
        season_ratio = np.clip(climo / max(climo.mean(), 0.1), 0.1, None)

        thresh = np.quantile(observed, 0.95)
        regime = observed >= max(thresh, 15.0)

        for lead in LEADS:
            base_noise_sd = 0.10 + 0.045 * lead
            err_sd = np.where(regime, base_noise_sd * (2.2 + 0.15 * lead), base_noise_sd)
            mult_noise = RNG.normal(1.0, err_sd)
            bias = np.where(regime, RNG.normal(-0.35, 0.2, n), 0.0)
            forecast = np.clip(observed * (mult_noise + bias), 0, None)

            true_uncertainty = err_sd * (climo + 1)
            spread_collapse = np.where(regime, RNG.uniform(0.35, 0.65, n), 1.0)
            ens_spread = np.clip(true_uncertainty * spread_collapse * RNG.normal(1.0, 0.15, n), 0.3, None)
            jumpiness = np.abs(RNG.normal(0, np.where(regime, 6.0, 1.2), n))
            moisture_anom = np.where(regime, RNG.normal(1.1, 0.4, n), RNG.normal(0, 0.5, n))
            shear_anom = np.where(regime, RNG.normal(-0.8, 0.3, n), RNG.normal(0, 0.5, n))

            rows.append(pd.DataFrame({
                "date": g["date"], "year": g["year"], "month": g["month"], "doy": g["doy"],
                "subdivision": sub, "lead_day": lead,
                "forecast_mm": forecast, "observed_mm": observed,
                "ens_spread": ens_spread, "jumpiness": jumpiness,
                "moisture_anom": moisture_anom, "shear_anom": shear_anom,
                "climatology_mm": climo, "regime_flag": regime.astype(int),
            }))

    df = pd.concat(rows, ignore_index=True)
    df.to_parquet("/manual/ps26079/data/dataset_hybrid.parquet", index=False)
    print(f"built {len(df):,} rows from REAL observed rainfall (2022-2025) + simulated forecast")
    print(f"regime-day fraction: {df['regime_flag'].mean():.3%}")
    return df


if __name__ == "__main__":
    build()
