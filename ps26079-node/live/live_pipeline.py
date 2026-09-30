import argparse
import json
import tempfile
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np

import live_core as core

HERE = Path(__file__).parent
INDIA = dict(lat_min=6.0, lat_max=39.0, lon_min=66.0, lon_max=100.0)


def _open_all(path):
    import cfgrib
    out = {}
    for ds in cfgrib.open_datasets(str(path), backend_kwargs={"indexpath": ""}):
        for name in ds.data_vars:
            out[name] = ds[name]
    return out


def _box_mean(da, lat_min, lat_max, lon_min, lon_max):
    da = da.sortby("latitude")
    return float(da.sel(latitude=slice(lat_min, lat_max), longitude=slice(lon_min, lon_max))
                 .mean(dim=["latitude", "longitude"], skipna=True).values)


def _client(source):
    from ecmwf.opendata import Client
    return Client(source=source, model="ifs")


def fetch_live_inputs(source: str, assets: dict):
    tmp = Path(tempfile.mkdtemp(prefix="ps26079_live_"))
    client = _client(source)

    latest = client.latest(type="fc", time=0, step=240, param="tp")
    init = latest.date()
    dstr = init.strftime("%Y-%m-%d")
    print(f"latest 00Z run: {dstr}")

    steps = [24 * L for L in core.LEADS]
    f_tp = tmp / "tp.grib2"
    client.retrieve(date=dstr, time=0, type="fc", step=steps, param="tp", target=str(f_tp))
    tp = _open_all(f_tp)["tp"]
    cum = {}
    for L in core.LEADS:
        field = tp.sel(step=np.timedelta64(24 * L, "h")) if "step" in tp.dims else tp
        cum[L] = _box_mean(field, INDIA["lat_min"], INDIA["lat_max"], INDIA["lon_min"], INDIA["lon_max"]) * 1000.0

    f_sfc, f_pl = tmp / "sfc.grib2", tmp / "pl.grib2"
    client.retrieve(date=dstr, time=0, type="fc", step=0, param=["2t", "2d", "msl"], target=str(f_sfc))
    client.retrieve(date=dstr, time=0, type="fc", step=0, param=["u", "v", "r"],
                    levtype="pl", levelist=[850, 200], target=str(f_pl))
    sfc, pl = _open_all(f_sfc), _open_all(f_pl)

    prev = (init - timedelta(days=1)).strftime("%Y-%m-%d")
    f_prev = tmp / "msl_prev.grib2"
    try:
        client.retrieve(date=prev, time=0, type="fc", step=0, param="msl", target=str(f_prev))
    except Exception:
        _client("aws").retrieve(date=prev, time=0, type="fc", step=0, param="msl", target=str(f_prev))
    msl_prev = _open_all(f_prev)["msl"]

    dd = sfc["t2m"] - sfc["d2m"]
    lev = "isobaricInhPa"
    shear = ((pl["u"].sel({lev: 850}) - pl["u"].sel({lev: 200})) ** 2 +
             (pl["v"].sel({lev: 850}) - pl["v"].sel({lev: 200})) ** 2) ** 0.5
    rh850 = pl["r"].sel({lev: 850})

    met = {}
    for sub, (la0, la1, lo0, lo1) in assets["boxes"].items():
        args = (la0, la1, lo0, lo1)
        met[sub] = {
            "dewpoint_depression": _box_mean(dd, *args),
            "wind_shear_850_200": _box_mean(shear, *args),
            "rh850": _box_mean(rh850, *args),
            "msl_tendency": _box_mean(sfc["msl"], *args) - _box_mean(msl_prev, *args),
        }
    return init, cum, met


def load_replay():
    r = json.loads((core.ASSETS_DIR / "replay_2024-08-20.json").read_text())
    init = date.fromisoformat(r["init_date"])
    cum = {int(k): v for k, v in r["india_tp_cumulative_mm"].items()}
    return init, cum, r["met"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="ecmwf", choices=["ecmwf", "aws", "azure"])
    ap.add_argument("--replay", action="store_true", help="use stored real historical inputs (no network)")
    ap.add_argument("--out", default=str(HERE / "live_predictions.json"))
    ap.add_argument("--node-data", default=None, help="folder to also copy live_predictions.json into")
    a = ap.parse_args()

    bundle = core.load_bundle()
    if a.replay:
        init, cum, met = load_replay()
        mode, extra = "replay", ["REPLAY: stored real inputs from a past date (2024-08-20), NOT a live forecast."]
    else:
        init, cum, met = fetch_live_inputs(a.source, bundle["assets"])
        mode, extra = "live", ["Atmospheric-state inputs come from ECMWF step-0 analysis fields, while the model "
                               "was trained on ERA5 (small source mismatch)."]

    daily = core.deaccumulate(cum)
    feats = core.build_features(init, daily, met, bundle["assets"])
    pred = core.predict(feats, bundle)
    payload = core.to_payload(pred, init, mode, daily,
                              datetime.now(timezone.utc).isoformat(timespec="seconds"), extra)

    text = json.dumps(payload, indent=1)
    Path(a.out).write_text(text)
    if a.node_data:
        (Path(a.node_data) / "live_predictions.json").write_text(text)
    top = pred.sort_values("bust_probability", ascending=False).head(5)
    print(f"[{mode}] init {init}  ->  wrote {a.out}")
    print(top[["subdivision", "lead_day", "valid_date", "bust_probability"]].to_string(index=False))
    for w in payload["warnings"]:
        print("WARNING:", w)


if __name__ == "__main__":
    main()
