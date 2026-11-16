"""GUM-style uncertainty of the corrected value: Monte-Carlo propagation of the lidar input uncertainties through the network, combined with
the model residual from the validation set. It also checks the empirical coverage of the 95 % interval on the synthetic test and "unseen" sets.
The input uncertainties (defaults below) are my ASSUMPTIONS, not measured values; adjust them for another instrument.
"""
import argparse, json, os
from windlidar.paths import rp
import numpy as np, pandas as pd
from windlidar.gustnet.model import load_predictor, split_indices

def mc_interval(pred, X, u_abs, u_rel, s_res, n_mc=500, seed=0):
    """X: (n, k) measured features. u_abs/u_rel: per-feature absolute and relative standard uncertainty."""
    rng = np.random.default_rng(seed); n, k = X.shape
    sd = np.sqrt(np.asarray(u_abs) ** 2 + (np.asarray(u_rel) * np.abs(X)) ** 2)          # (n, k)
    draws = X[None] + rng.standard_normal((n_mc, n, k)) * sd[None]
    P = pred.predict(draws.reshape(-1, k)).reshape(n_mc, n)
    mean = P.mean(0); u_mc = P.std(0); u_c = np.sqrt(u_mc ** 2 + s_res ** 2)
    return mean, u_c, u_mc

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="full", choices=["full", "level2"])
    ap.add_argument("--models", default=rp("models/baseline")); ap.add_argument("--results", default=rp("results/gustnet"))
    ap.add_argument("--data", default=rp("data/dataset.csv"))
    ap.add_argument("--u-speed", type=float, default=0.10, help="abs. std uncertainty of lidar mean wind [m/s]")
    ap.add_argument("--u-speed-rel", type=float, default=0.01)
    ap.add_argument("--u-ti-rel", type=float, default=0.05, help="relative std uncertainty of lidar TI")
    ap.add_argument("--u-gf-rel", type=float, default=0.01, help="relative std uncertainty of lidar gust factor")
    ap.add_argument("--n-mc", type=int, default=500)
    a = ap.parse_args()
    pred = load_predictor(os.path.join(a.models, "best_%s.pt" % a.features))
    met = json.load(open(os.path.join(a.results, "metrics_%s.json" % a.features))); s_res = met["val_residual_std"]
    u_abs = {"U_lidar": a.u_speed, "TI_lidar": 0.0, "GF_lidar": 0.0}; u_rel = {"U_lidar": a.u_speed_rel, "TI_lidar": a.u_ti_rel, "GF_lidar": a.u_gf_rel}
    ua = [u_abs[f] for f in pred.features]; ur = [u_rel[f] for f in pred.features]
    df = pd.read_csv(a.data); tr, va, te, oo = split_indices(df, 0); out = {"assumptions": dict(u_abs=dict(zip(pred.features, ua)), u_rel=dict(zip(pred.features, ur)), model_residual_std=s_res)}
    for name, ii in (("test", te), ("unseen", oo)):
        X = df[pred.features].values[ii]; y = df[pred.target].values[ii]
        mean, u_c, u_mc = mc_interval(pred, X, ua, ur, s_res, a.n_mc)
        cover = float(np.mean(np.abs(y - mean) <= 1.96 * u_c))
        out[name] = dict(coverage_95=cover, mean_halfwidth=float(np.mean(1.96 * u_c)), mean_u_mc=float(np.mean(u_mc)), n=len(ii))
        print(name, out[name])
    json.dump(out, open(os.path.join(a.results, "uncertainty_%s.json" % a.features), "w"), indent=1)

if __name__ == "__main__":
    main()
