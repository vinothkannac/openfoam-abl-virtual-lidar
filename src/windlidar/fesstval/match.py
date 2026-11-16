"""Build a quality-filtered, time-matched sonic/lidar table from the FESSTVaL level-2 files (Falkenberg, about 90 m).

Sonic: falkenberg_sonic90m with qc_wind == 1. Lidar: the DWD product (qwind == 1, gate 92.7 m) or the Cologne/UzK product (gate 90.3 m),
at the gate closest to the sonic height. Both files use the same 10-min grid, so the match is exact. Periods with a mean wind below 3 m/s are
dropped because gust factors are not meaningful there. Both "gust" values are 3 s maxima within 10 min, so the definitions agree.
"""
import argparse, glob, os
from windlidar.paths import rp
import numpy as np, pandas as pd, xarray as xr


def load_sonic(root):
    rows = []
    for f in sorted(glob.glob(os.path.join(root, "falkenberg_sonic90m/level2/*.nc"))):
        d = xr.open_dataset(f)
        rows.append(pd.DataFrame({"t": d.time.values, "mean_s": d.wspeed.values, "gust_s": d.wspeed_max.values,
                                  "wdir": d.wdir.values, "qc_s": d.qc_wind.values}))
    S = pd.concat(rows).sort_values("t")
    return S[S.qc_s == 1].drop(columns="qc_s")


def load_lidar(root, product, z=90.3):
    rows = []
    for f in sorted(glob.glob(os.path.join(root, f"falkenberg_dlidcsm/level2{product}/*/*.nc"))):
        d = xr.open_dataset(f)
        j = int(np.argmin(np.abs(d.height.values - z)))
        r = {"t": d.time.values, "mean_l": d.wspeed.isel(height=j).values, "gust_l": d.wspeed_max.isel(height=j).values,
             "z_l": float(d.height.values[j])}
        if product == "dwd":
            r["q_l"] = d.qwind.isel(height=j).values
            r["good_l"] = d.n_good_data.isel(height=j).values
            r["acf_l"] = d.acf.isel(height=j).values
        else:
            r["sd_l"] = d.sd_wspeed.isel(height=j).values
            r["q_l"] = np.ones(len(d.time))
        rows.append(pd.DataFrame(r))
    L = pd.concat(rows).sort_values("t")
    return L[L.q_l == 1]


def build(root, product, min_wind=3.0):
    S, L = load_sonic(root), load_lidar(root, product)
    M = pd.merge(S, L, on="t", how="inner").dropna()          # both are on the same 10-min grid -> exact match
    M = M[(M.mean_s >= min_wind) & (M.mean_l >= min_wind)].copy()
    M["gf_s"] = M.gust_s / M.mean_s
    M["gf_l"] = M.gust_l / M.mean_l
    M["hour"] = pd.DatetimeIndex(M.t).hour
    return M.reset_index(drop=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=rp("real_data"))
    ap.add_argument("--out", default=rp("results/fesstval"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    for p in ("dwd", "uzk"):
        M = build(a.root, p)
        M.to_csv(os.path.join(a.out, f"matched_{p}.csv"), index=False)
        print(p, len(M), "periods", M.t.min(), "->", M.t.max(), " lidar gate z=%.1f m" % M.z_l.iloc[0])
