"""Calibrate the stability emulator parameters (H_day, H_night, d, noise, ...) against the REAL TRAINING PERIOD.

I match summary statistics (day/night gust-factor means, spread, lidar-minus-sonic bias, correlation) with a random search using common
random numbers, followed by local refinement, parallelised over CPU cores. The test period (>= --split) is never touched, so the calibration
cannot leak test information. Caveat: several fitted parameters end up at their bounds, so part of the fit compensates for missing physics.
"""
import argparse, json, os
from windlidar.paths import rp
import numpy as np, pandas as pd
from multiprocessing import Pool
from windlidar.gustnet.dataset_stability import generate

DAY = (9, 17)
KEYS = ["bias", "sd_diff", "corr", "gfs_day", "gfs_night", "gfl_day", "gfl_night", "sd_gfs", "sd_gfl"]
SCALE = dict(bias=0.015, sd_diff=0.015, corr=0.04, gfs_day=0.02, gfs_night=0.02, gfl_day=0.02, gfl_night=0.02, sd_gfs=0.015, sd_gfl=0.015)


def moments(gs, gl, hour):
    day = (hour >= DAY[0]) & (hour <= DAY[1]); d = gl - gs
    return dict(bias=d.mean(), sd_diff=d.std(), corr=np.corrcoef(gs, gl)[0, 1], gfs_day=gs[day].mean(), gfs_night=gs[~day].mean(),
                gfl_day=gl[day].mean(), gfl_night=gl[~day].mean(), sd_gfs=gs.std(), sd_gfl=gl.std())


def loss_of(sim, real):
    return float(sum(((sim[k] - real[k]) / SCALE[k]) ** 2 for k in KEYS))


def evaluate(args):
    th, hours, real, n, seed = args
    P = dict(H_day=th[0], H_night=th[1], d=th[2], noise=th[3], sig_spread=th[4], noise_rel=th[5])
    df = generate(n, P, hours, seed=seed)
    m = moments(df.GF_sonic.values, df.GF_lidar.values, df.hour.values)
    return th, loss_of(m, real), m


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--matched", default=rp("results/fesstval/matched_dwd.csv"))
    ap.add_argument("--split", default="2021-07-15")
    ap.add_argument("--n", type=int, default=1500); ap.add_argument("--n-random", type=int, default=96); ap.add_argument("--n-refine", type=int, default=64)
    ap.add_argument("--out", default=rp("results/stability_emulator_params.json"))
    a = ap.parse_args()
    R = pd.read_csv(a.matched, parse_dates=["t"]); R = R[R.t < a.split]
    real = moments(R.gf_s.values, R.gf_l.values, R.hour.values); hours = R.hour.values
    print("real training-period moments:", {k: round(float(v), 3) for k, v in real.items()}, "n =", len(R), flush=True)
    rng = np.random.default_rng(0)
    lo, hi = np.array([0.02, 0.0, 0.0, 0.0, 0.05, 0.0]), np.array([0.6, 0.15, 600.0, 0.6, 0.40, 0.8])
    pool = Pool(16)
    cand = [lo + (hi - lo) * rng.random(6) for _ in range(a.n_random)]
    res = pool.map(evaluate, [(c, hours, real, a.n, 7) for c in cand])
    res.sort(key=lambda r: r[1]); print("random search best loss %.1f  %s" % (res[0][1], np.round(res[0][0], 3)), flush=True)
    top = [r[0] for r in res[:5]]; span = (hi - lo) * 0.12
    cand = [np.clip(top[rng.integers(5)] + span * rng.normal(size=6), lo, hi) for _ in range(a.n_refine)]
    res2 = pool.map(evaluate, [(c, hours, real, a.n, 7) for c in cand]); res = sorted(res + res2, key=lambda r: r[1])
    # re-score the best few with a different seed and more cases to avoid picking a lucky draw
    fin = pool.map(evaluate, [(r[0], hours, real, 4000, 11) for r in res[:6]]); fin.sort(key=lambda r: r[1])
    th, L, m = fin[0]
    out = dict(params=dict(H_day=float(th[0]), H_night=float(th[1]), d=float(th[2]), noise=float(th[3]), sig_spread=float(th[4]), noise_rel=float(th[5])), loss=L,
               real_moments={k: float(v) for k, v in real.items()}, sim_moments={k: float(v) for k, v in m.items()},
               calibrated_on="chronological training period only (< %s), n=%d" % (a.split, len(R)),
               top_runners_up=[dict(theta=[float(x) for x in r[0]], loss=r[1]) for r in fin[1:]])
    json.dump(out, open(a.out, "w"), indent=1)
    print("BEST", out["params"], "loss %.1f" % L)
    for k in KEYS: print("  %-10s real %.3f  sim %.3f" % (k, real[k], m[k]))
