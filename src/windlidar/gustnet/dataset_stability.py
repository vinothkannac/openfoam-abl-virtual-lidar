"""Synthetic dataset from the stability-aware emulator (atmosphere/stability.py).

One row = one 10-min record seen by the sonic and by the (decorrelated) lidar. The hours of day are resampled from real training-period hours
so that day/night frequencies match the campaign. Usage: python -m windlidar.gustnet.dataset_stability --n 8000
"""
import argparse, json
from windlidar.paths import rp
import numpy as np, pandas as pd
from windlidar.atmosphere.stability import sample_state, synth_pair
from windlidar.lidar.gust_proxy import sonic_stats, lidar_stats

DEFAULT = dict(H_day=0.15, H_night=0.03, d=150.0, noise=0.4)


def generate(n, P, hours, seed=0, z=90.0, Ts=3.4, leff_range=(40.0, 150.0), tag=0):
    rng = np.random.default_rng(seed * 100003 + tag)
    rows = []
    while len(rows) < n:
        h = int(rng.choice(hours))
        st = sample_state(rng, h, z, P)
        if st["U"] < 2.5 or st["U"] > 16:
            continue
        d = P["d"] * rng.lognormal(0, 0.3) if P["d"] > 0 else 0.0
        us, vs, ul, vl = synth_pair(st["U"], st["sigma_u"], st["L_u"], d, rng=rng)
        s = sonic_stats(st["U"], us, vs)
        l = lidar_stats(st["U"], ul, vl, rng.uniform(*leff_range), Ts=Ts, noise_std=(P["noise"] + P.get("noise_rel", 0.0) * st["sigma_u"]) * rng.uniform(0.6, 1.4), rng=rng)
        if l["mean"] < 3.0 or s["mean"] < 3.0:        # same 3 m/s filter as the real data
            continue
        rows.append(dict(hour=h, U_lidar=l["mean"], GF_lidar=l["gf"], TI_lidar=l["ti"], U_sonic=s["mean"], GF_sonic=s["gf"], TI_sonic=s["ti"],
                         ustar=st["ustar"], zeta=st["zeta"], Hs=st["Hs"], L_eff=0.0, sigma_ratio=st["sigma_ratio"]))
    df = pd.DataFrame(rows)
    df["sin_h"] = np.sin(2 * np.pi * df.hour / 24); df["cos_h"] = np.cos(2 * np.pi * df.hour / 24)
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", default=None)
    ap.add_argument("--hours-from", default=rp("results/fesstval/matched_dwd.csv"))
    ap.add_argument("--before", default="2021-07-15", help="only hours of real periods before this date are used (training period)")
    ap.add_argument("--n", type=int, default=8000); ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default=rp("data/dataset_stability.csv"))
    a = ap.parse_args()
    P = json.load(open(a.params))["params"] if a.params else DEFAULT
    R = pd.read_csv(a.hours_from, parse_dates=["t"]); hours = R[R.t < a.before].hour.values
    df = generate(a.n, P, hours, seed=a.seed)
    df.to_csv(a.out, index=False); print(a.out, len(df), P)
    print(df[["U_lidar", "GF_lidar", "GF_sonic", "zeta"]].describe().loc[["mean", "std", "min", "max"]].round(3))
