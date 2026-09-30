import json
import pandas as pd

scored = pd.read_parquet("/manual/ps26079/outputs/scored_test_era5.parquet")
scored["date"] = pd.to_datetime(scored["date"])

daily_bust_rate = scored.groupby("date")["bust"].mean()
high_bust_dates = daily_bust_rate.sort_values(ascending=False).head(8).index
other_dates = daily_bust_rate[daily_bust_rate == 0].sample(7, random_state=1).index
demo_dates = sorted(list(high_bust_dates) + list(other_dates))

grid = {}
for d in demo_dates:
    day_df = scored[scored["date"] == d]
    subs = {}
    for sub, g in day_df.groupby("subdivision"):
        g = g.sort_values("lead_day")
        subs[sub] = [
            {"lead_day": int(r.lead_day), "p": round(float(r.bust_probability), 3),
             "forecast_mm": round(float(r.forecast_mm), 1), "regime": int(r.bust)}
            for r in g.itertuples()
        ]
    grid[str(d.date())] = subs

with open("/manual/ps26079/outputs/review_queue_era5.json") as f:
    review_queue = json.load(f)
with open("/manual/ps26079/outputs/era5_results.json") as f:
    eval_results_raw = json.load(f)
eval_results = {"results": eval_results_raw["results"]}

from ingest_real_imd import SUBDIVISION_BOXES
centroids = {sub: [round((la+lb)/2,2), round((oa+ob)/2,2)] for sub,(la,lb,oa,ob) in SUBDIVISION_BOXES.items()}

bundle = {"grid": grid, "review_queue": review_queue, "eval_results": eval_results,
          "subdivisions": sorted(scored["subdivision"].unique().tolist()),
          "dates": list(grid.keys()), "centroids": centroids}
with open("/manual/ps26079/outputs/dashboard_bundle_era5.json", "w") as f:
    json.dump(bundle, f)

print(f"dates: {len(grid)}, subdivisions: {len(bundle['subdivisions'])}")
print(f"bundle size: {len(json.dumps(bundle))/1024:.1f} KB")
