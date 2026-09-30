import sys
from datetime import date
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import live_core as core
import live_pipeline as lp

B = core.load_bundle()


def test_deaccumulate():
    assert core.deaccumulate({1: 5.0, 2: 8.0, 3: 8.5}) == {1: 5.0, 2: 3.0, 3: 0.5}
    assert core.deaccumulate({1: 5.0, 2: 4.99})[2] == 0.0


def test_downscale_and_features():
    init, cum, met = lp.load_replay()
    daily = core.deaccumulate(cum)
    df = core.build_features(init, daily, met, B["assets"])
    r = df[(df.subdivision == "Kerala") & (df.lead_day == 3)].iloc[0]
    m = str((init + pd.Timedelta(days=3)).month)
    assert abs(r.forecast_mm - daily[3] * B["assets"]["ratio"]["Kerala"][m]) < 1e-9
    assert abs(r.climatology_anom - (r.forecast_mm - B["assets"]["sub_climo"]["Kerala"][m])) < 1e-9
    assert len(df) == 36 * 10 and list(B["features"]) == [c for c in B["features"] if c in df.columns]


def test_replay_matches_training_predictions():
    init, cum, met = lp.load_replay()
    df = core.build_features(init, core.deaccumulate(cum), met, B["assets"])
    pred = core.predict(df, B)
    ref = pd.read_parquet(Path(__file__).resolve().parent / "reference_scored.parquet")
    pass
    m = pred.merge(ref[["subdivision", "lead_day", "valid_date", "bust_probability"]],
                   on=["subdivision", "lead_day", "valid_date"], suffixes=("", "_ref"))
    assert len(m) == 360
    assert np.abs(m.bust_probability - m.bust_probability_ref).max() < 1e-3


def test_outputs_valid():
    init, cum, met = lp.load_replay()
    pred = core.predict(core.build_features(init, core.deaccumulate(cum), met, B["assets"]), B)
    assert pred.bust_probability.between(0, 1).all()
    assert np.allclose(pred.confidence, 1 - pred.bust_probability)
    assert all(len(r) >= 1 for r in pred.reasons)


def test_out_of_season_flag():
    init, cum, met = lp.load_replay()
    late = date(2026, 9, 28)
    pred = core.predict(core.build_features(late, core.deaccumulate(cum), met, B["assets"]), B)
    assert not pred[pred.lead_day == 1].out_of_season.any()
    assert pred[pred.lead_day == 5].out_of_season.all()
    payload = core.to_payload(pred, late, "live", core.deaccumulate(cum), "t")
    assert any("outside Jun-Sep" in w for w in payload["warnings"])
