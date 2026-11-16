"""Why the synthetic-trained correction fails on real data, and what a real-data recalibration achieves.

I compare a ladder of models for the sonic gust factor at about 90 m on a chronological hold-out (train before --split, test after) and with
6-fold blocked cross-validation. Chronological and blocked splits are deliberate: neighbouring 10-min records are strongly autocorrelated, so a
random split would leak information and inflate the scores.
  raw           lidar gust factor unchanged
  synthetic_nn  network trained on synthetic data only (the original method; no real target values used)
  lin_gf        linear recalibration fitted on the real training period
  lin_gf_U      + mean wind
  lin_diurnal   + sin/cos(hour), a cheap stability proxy that needs no extra instrument
  gbr_diurnal   gradient boosting on U, GF, sin/cos(hour)
"""
import argparse, json, os
from windlidar.paths import rp
import numpy as np, pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from windlidar.gustnet.model import load_predictor


def metrics(y, p):
    e = p - y
    return dict(bias=float(e.mean()), MAE=float(np.abs(e).mean()), RMSE=float(np.sqrt((e ** 2).mean())),
                R2=float(1 - (e ** 2).sum() / ((y - y.mean()) ** 2).sum()), n=int(len(y)))


def feats(M, kind):
    h = 2 * np.pi * M.hour.values / 24
    cols = {"gf": [M.gf_l.values], "gf_U": [M.gf_l.values, M.mean_l.values],
            "diurnal": [M.gf_l.values, M.mean_l.values, np.sin(h), np.cos(h)]}[kind]
    return np.column_stack(cols)


def models(pred):
    return {
        "raw": (None, None),
        "synthetic_nn": ("nn", None),
        "lin_gf": ("gf", lambda: LinearRegression()),
        "lin_gf_U": ("gf_U", lambda: LinearRegression()),
        "lin_diurnal": ("diurnal", lambda: LinearRegression()),
        "gbr_diurnal": ("diurnal", lambda: GradientBoostingRegressor(n_estimators=150, max_depth=3, learning_rate=0.05, subsample=0.8, random_state=0)),
    }


def fit_predict(name, spec, tr, te, pred):
    kind, mk = spec
    if name == "raw":
        return te.gf_l.values
    if name == "synthetic_nn":
        col = {"U_lidar": te.mean_l.values, "GF_lidar": te.gf_l.values,
               "sin_h": np.sin(2 * np.pi * te.hour.values / 24), "cos_h": np.cos(2 * np.pi * te.hour.values / 24)}
        return pred.predict(np.column_stack([col[f] for f in pred.features]))
    m = mk().fit(feats(tr, kind), tr.gf_s.values)
    return m.predict(feats(te, kind))


def run(M, split, model_path):
    pred = load_predictor(model_path)
    specs = models(pred)
    M = M.sort_values("t").reset_index(drop=True)
    tr, te = M[M.t < split], M[M.t >= split]
    out = {"n_train": len(tr), "n_test": len(te), "holdout": {}, "holdout_day": {}, "holdout_night": {}, "blocked_cv": {}}
    day = te.hour.between(9, 17).values
    P = {}
    for n, s in specs.items():
        P[n] = fit_predict(n, s, tr, te, pred)
        out["holdout"][n] = metrics(te.gf_s.values, P[n])
        out["holdout_day"][n] = metrics(te.gf_s.values[day], P[n][day])
        out["holdout_night"][n] = metrics(te.gf_s.values[~day], P[n][~day])
    # blocked CV: 6 contiguous folds in time (no random shuffling -> no leakage from autocorrelated neighbours)
    folds = np.array_split(np.arange(len(M)), 6)
    cv = {n: [] for n in specs}
    for f in folds:
        te_f, tr_f = M.iloc[f], M.drop(M.index[f])
        for n, s in specs.items():
            cv[n].append(metrics(te_f.gf_s.values, fit_predict(n, s, tr_f, te_f, pred)))
    for n in specs:
        out["blocked_cv"][n] = {k: [float(np.mean([c[k] for c in cv[n]])), float(np.std([c[k] for c in cv[n]]))] for k in ("RMSE", "R2", "bias")}
    return out, te, P


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matched", default=rp("results/fesstval"))
    ap.add_argument("--model", default=rp("models/baseline/best_level2.pt"))
    ap.add_argument("--split", default="2021-07-15")
    a = ap.parse_args()
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    allres = {}
    for p in ("dwd", "uzk"):
        M = pd.read_csv(os.path.join(a.matched, f"matched_{p}.csv"), parse_dates=["t"])
        res, te, P = run(M, pd.Timestamp(a.split), a.model)
        allres[p] = res
        print(f"\n=== {p}: train {res['n_train']}  test {res['n_test']} (from {a.split})")
        print(f"{'model':14s} {'bias':>7s} {'RMSE':>7s} {'R2':>6s} | day RMSE night RMSE | CV RMSE (±sd)   CV R2")
        for n in res["holdout"]:
            h, d, ni, c = res["holdout"][n], res["holdout_day"][n], res["holdout_night"][n], res["blocked_cv"][n]
            print(f"{n:14s} {h['bias']:7.3f} {h['RMSE']:7.3f} {h['R2']:6.3f} | {d['RMSE']:8.3f} {ni['RMSE']:10.3f} | {c['RMSE'][0]:.3f} ±{c['RMSE'][1]:.3f}  {c['R2'][0]:.3f}")
        fig, ax = plt.subplots(1, 3, figsize=(14, 4.2))
        for a_, n in zip(ax[:2], ("raw", "lin_diurnal")):
            a_.scatter(te.gf_s, P[n], s=5, alpha=.35); lo, hi = te.gf_s.min(), te.gf_s.max(); a_.plot([lo, hi], [lo, hi], "k--")
            a_.set_title(f"{p.upper()}  {n}  R2={res['holdout'][n]['R2']:.2f}"); a_.set_xlabel("sonic GF"); a_.set_ylabel("predicted GF")
        g = te.groupby("hour").apply(lambda d: pd.Series({"sonic": d.gf_s.mean(), "lidar": d.gf_l.mean()}), include_groups=False)
        g.plot(ax=ax[2]); ax[2].set_title("mean GF by hour of day (UTC)"); ax[2].set_xlabel("hour")
        fig.tight_layout(); fig.savefig(os.path.join(a.matched, f"real_analysis_{p}.png"), dpi=130)
    json.dump(allres, open(os.path.join(a.matched, "real_analysis.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
