"""First, deliberately naive comparison of the synthetic-trained network with real FESSTVaL data (nearest-time matching, no quality flags).

I kept this script because it documents the initial attempt: raw lidar R2 0.26 versus corrected 0.22. The quality-controlled evaluation is in
fesstval/match.py and fesstval/analysis.py. Use fesstval/inspect_netcdf.py to find variable names if your files differ.
"""
import argparse, json, os
from windlidar.paths import rp
import numpy as np, pandas as pd, xarray as xr
from windlidar.gustnet.model import load_predictor

def series(ds, var, time_var, index):
    v = ds[var]
    for d in v.dims:
        if d != time_var:
            v = v.isel({d: index})
    return pd.Series(np.asarray(v.values, dtype=float), index=pd.DatetimeIndex(ds[time_var].values), name=var)

def speed(ds, speed_var, u_var, v_var, time_var, index):
    if speed_var:
        return series(ds, speed_var, time_var, index)
    return np.hypot(series(ds, u_var, time_var, index), series(ds, v_var, time_var, index))

def metrics(y, p):
    e = p - y
    return dict(bias=float(e.mean()), MAE=float(np.abs(e).mean()), RMSE=float(np.sqrt((e ** 2).mean())),
                R2=float(1 - (e ** 2).sum() / ((y - y.mean()) ** 2).sum()), n=int(len(y)))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sonic", nargs="+", required=True, help="sonic level-2 netCDF file(s)")
    ap.add_argument("--lidar", nargs="+", required=True, help="lidar level-2 netCDF file(s)")
    ap.add_argument("--model", default=rp("models/baseline/best_level2.pt"))
    ap.add_argument("--time-var", default="time")
    ap.add_argument("--sonic-gust-var", default="gust"); ap.add_argument("--sonic-speed-var", default="wind_speed")
    ap.add_argument("--sonic-u-var", default=None); ap.add_argument("--sonic-v-var", default=None)
    ap.add_argument("--sonic-index", type=int, default=2, help="index along the height dimension (2 = 90.3 m if heights are 2.4/50.3/90.3)")
    ap.add_argument("--lidar-gust-var", default="gust"); ap.add_argument("--lidar-speed-var", default=None)
    ap.add_argument("--lidar-u-var", default="u"); ap.add_argument("--lidar-v-var", default="v")
    ap.add_argument("--lidar-index", type=int, default=0, help="index along height dimension of the lidar product")
    ap.add_argument("--min-wind", type=float, default=3.0, help="discard 10-min periods with mean wind below this [m/s]")
    ap.add_argument("--tolerance-min", type=float, default=5.0)
    ap.add_argument("--out", default=rp("results/gustnet"))
    a = ap.parse_args()

    def load(files, kind):
        dss = [xr.open_dataset(f, decode_times=True) for f in files]
        return dss
    sd = load(a.sonic, "sonic"); ld = load(a.lidar, "lidar")
    S = pd.concat([pd.DataFrame({"gust_s": series(d, a.sonic_gust_var, a.time_var, a.sonic_index),
                                 "mean_s": speed(d, a.sonic_speed_var if not a.sonic_u_var else None, a.sonic_u_var, a.sonic_v_var, a.time_var, a.sonic_index)}) for d in sd]).sort_index()
    L = pd.concat([pd.DataFrame({"gust_l": series(d, a.lidar_gust_var, a.time_var, a.lidar_index),
                                 "mean_l": speed(d, a.lidar_speed_var, a.lidar_u_var, a.lidar_v_var, a.time_var, a.lidar_index)}) for d in ld]).sort_index()
    M = pd.merge_asof(S.reset_index().rename(columns={"index": "t"}), L.reset_index().rename(columns={"index": "t"}), on="t",
                      direction="nearest", tolerance=pd.Timedelta(minutes=a.tolerance_min)).dropna()
    M = M[(M.mean_s > a.min_wind) & (M.mean_l > a.min_wind) & np.isfinite(M.gust_s) & np.isfinite(M.gust_l)]
    if len(M) < 10:
        raise SystemExit("only %d matched periods - check time overlap, variable names, units" % len(M))
    M["gf_s"] = M.gust_s / M.mean_s; M["gf_l"] = M.gust_l / M.mean_l
    pred = load_predictor(a.model)
    M["gf_corr"] = pred.predict(M[["mean_l", "gf_l"]].values)
    res = dict(raw_lidar=metrics(M.gf_s.values, M.gf_l.values), corrected=metrics(M.gf_s.values, M.gf_corr.values),
               n_periods=len(M), mean_sonic_GF=float(M.gf_s.mean()), mean_raw_GF=float(M.gf_l.mean()), mean_corrected_GF=float(M.gf_corr.mean()))
    print(json.dumps(res, indent=1))
    os.makedirs(a.out, exist_ok=True)
    json.dump(res, open(os.path.join(a.out, "fesstval_comparison.json"), "w"), indent=1)
    M.to_csv(os.path.join(a.out, "fesstval_matched.csv"), index=False)
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(10, 4.3))
    ax[0].scatter(M.gf_s, M.gf_l, s=8, alpha=.5, label="raw lidar"); ax[0].scatter(M.gf_s, M.gf_corr, s=8, alpha=.5, label="corrected")
    lo, hi = M.gf_s.min(), M.gf_s.max(); ax[0].plot([lo, hi], [lo, hi], "k--"); ax[0].set_xlabel("sonic gust factor (90.3 m)"); ax[0].set_ylabel("lidar gust factor"); ax[0].legend()
    ax[1].hist([M.gf_l - M.gf_s, M.gf_corr - M.gf_s], bins=30, label=["raw", "corrected"]); ax[1].set_xlabel("GF error vs sonic"); ax[1].legend()
    fig.tight_layout(); fig.savefig(os.path.join(a.out, "fesstval_comparison.png"), dpi=130)

if __name__ == "__main__":
    main()
