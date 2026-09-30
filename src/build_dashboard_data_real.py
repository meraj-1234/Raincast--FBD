import json
import pandas as pd
import numpy as np

scored = pd.read_parquet("/manual/ps26079/outputs/scored_test_real.parquet")
scored["date"] = pd.to_datetime(scored["date"])

daily_regime_rate = scored.groupby("date")["regime_flag"].mean()
high_regime_dates = daily_regime_rate.sort_values(ascending=False).head(8).index
quiet_dates = daily_regime_rate[daily_regime_rate == 0].sample(7, random_state=1).index
demo_dates = sorted(list(high_regime_dates) + list(quiet_dates))

grid = {}
for d in demo_dates:
    day_df = scored[scored["date"] == d]
    subs = {}
    for sub, g in day_df.groupby("subdivision"):
        g = g.sort_values("lead_day")
        subs[sub] = [
            {
                "lead_day": int(r.lead_day),
                "p": round(float(r.bust_probability), 3),
                "forecast_mm": round(float(r.forecast_mm), 1),
                "regime": int(r.regime_flag),
            }
            for r in g.itertuples()
        ]
    grid[str(d.date())] = subs

with open("/manual/ps26079/outputs/dashboard_grid_real.json", "w") as f:
    json.dump(grid, f)

with open("/manual/ps26079/outputs/review_queue_real.json") as f:
    review_queue = json.load(f)

with open("/manual/ps26079/outputs/eval_results_real.json") as f:
    eval_results = json.load(f)

bundle = {"grid": grid, "review_queue": review_queue, "eval_results": eval_results,
          "subdivisions": sorted(scored["subdivision"].unique().tolist()),
          "dates": list(grid.keys())}
with open("/manual/ps26079/outputs/dashboard_bundle_real.json", "w") as f:
    json.dump(bundle, f)

print(f"dates: {len(grid)}, subdivisions: {len(bundle['subdivisions'])}")
print(f"bundle size: {len(json.dumps(bundle))/1024:.1f} KB")
