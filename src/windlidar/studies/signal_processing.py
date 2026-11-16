"""Signal-processing study with the virtual lidar.

I wanted to see what the scanning geometry does to the wind signal before building any correction. Three questions:
  1. the empirical lidar transfer function T(f) = S_lidar/S_sonic, compared with the boxcar model assumed in lidar/gust_proxy.py
  2. can the retrieval-noise floor be estimated blindly from the high-frequency end of the lidar spectrum? (it cannot, see README)
  3. correction of the turbulence intensity: raw, noise-subtracted, and noise-subtracted plus a data-driven attenuation ratio R(U)
Train/test split is by record. Usage: python -m windlidar.studies.signal_processing --n 800
"""
import argparse, json, os, sys
from windlidar.paths import rp
import numpy as np
from multiprocessing import Pool
from scipy.signal import welch
from windlidar.lidar import virtual as V

NV, PTS, NS = V.geometry()
TS = 3.4
AH = np.linalg.pinv(NV)


def one(seed):
    rng = np.random.default_rng(seed)
    U = rng.uniform(4, 14); ti = rng.uniform(0.07, 0.20); Lu = 340.0 * rng.lognormal(0, 0.3); sr = rng.uniform(0.1, 0.4)
    F = V.field(U, ti * U, Lu, PTS, rng)
    vr = V.scan(F, U, NV, NS, Ts=TS, sigma_r=sr, rng=rng)
    u, v, w = V.retrieve(vr, NV)
    sl = V.lidar_stats(U + u * 0 + u, v)["series"] if False else np.hypot(u, v)
    ss = V.sonic_gust(F, U); s_son = np.hypot(U + F[0, 0], F[1, 0])
    fl, Pl = welch(sl, fs=1 / TS, nperseg=32, noverlap=16, detrend="linear")
    fs_, Ps = welch(s_son, fs=V.FS, nperseg=1090, noverlap=545, detrend="linear")
    # exact noise std of the retrieved horizontal speed from the LS covariance (~ along-wind component)
    sig_n = sr * np.sqrt((np.linalg.inv(NV.T @ NV))[0, 0])
    return dict(U=U, ti_par=ti, Lu=Lu, sr=sr, sig_n=sig_n, U_l=sl.mean(), sig_l=sl.std(), gf_l=sl.max() / sl.mean(), U_s=ss["mean"], sig_s=ss["sigma"], ti_s=ss["ti"], gf_s=ss["gf"],
                Pl=Pl[:16], Ps=Ps[:16], fl=fl[:16])


def lin_fit(X, y, deg=3):
    return np.polyfit(X, y, deg)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=800); ap.add_argument("--out", default=rp("results/analysis"))
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    with Pool(16) as p:
        R = p.map(one, range(a.n))
    g = lambda k: np.array([r[k] for r in R])
    U, U_l, sig_l, sig_s, ti_s, ti_par = g("U"), g("U_l"), g("sig_l"), g("sig_s"), g("ti_s"), g("ti_par")
    sig_n, gf_s, gf_l = g("sig_n"), g("gf_s"), g("gf_l")
    Pl, Ps, fl = np.array([r["Pl"] for r in R]), np.array([r["Ps"] for r in R]), R[0]["fl"]
    out = {"n": a.n}
    # (1) empirical transfer function in 3 wind-speed bins vs boxcar model sinc^2(f Leff/U) sinc^2(f Ts)
    bins = [(4, 7), (7, 10), (10, 14)]; T_emp = {}
    for lo, hi in bins:
        m = (U >= lo) & (U < hi) & (fl[1:, None].shape[0] > 0)
        T = (Pl[m] - 2 * TS * (sig_n[m, None] ** 2)).mean(0) / Ps[m].mean(0)        # noise-subtracted
        T_emp[f"{lo}-{hi}"] = T.tolist()
        Ls = np.arange(10, 400, 5.0); Um = U[m].mean()
        err = [np.sum((np.log(np.clip(T[1:12], 1e-3, None)) - np.log(np.sinc(fl[1:12] * L / Um) ** 2 * np.sinc(fl[1:12] * TS) ** 2 + 1e-9)) ** 2) for L in Ls]
        out.setdefault("best_Leff_boxcar_m", {})[f"{lo}-{hi}"] = float(Ls[int(np.argmin(err))])
    out["T_emp"] = T_emp; out["freq"] = fl.tolist()
    # (2) blind noise-floor estimate: median PSD in top band (f > 0.115 Hz) assuming noise is white (PSD = 2 sig_n^2 Ts); turbulence leakage there biases it high
    top = fl > 0.115
    sn2_hat = np.median(Pl[:, top], axis=1) / (2 * TS)
    out["noise_floor"] = dict(median_ratio_est_over_true_var=float(np.median(sn2_hat / sig_n ** 2)), p10=float(np.percentile(sn2_hat / sig_n ** 2, 10)), p90=float(np.percentile(sn2_hat / sig_n ** 2, 90)))
    # (3) TI correction, train/test by record
    idx = np.arange(a.n); tr, te = idx % 2 == 0, idx % 2 == 1
    def metrics(est, ref):
        e = est - ref; return dict(bias=float(e.mean()), RMSE=float(np.sqrt((e ** 2).mean())), R2=float(1 - (e ** 2).sum() / ((ref - ref.mean()) ** 2).sum()))
    ti_raw = sig_l / U_l
    var_ns = np.clip(sig_l ** 2 - sn2_hat, 1e-4, None)                       # blind noise-subtracted variance
    ti_ns = np.sqrt(var_ns) / U_l
    # data-driven attenuation ratio R(U) = var_nosie-free_lidar / var_sonic as a smooth function of U, fitted on the train half
    ratio = var_ns / sig_s ** 2
    c = lin_fit(U_l[tr], ratio[tr], 3)
    Rhat = np.clip(np.polyval(c, U_l), 0.1, 1.5)
    ti_full = np.sqrt(var_ns / Rhat) / U_l
    out["TI_test"] = dict(raw=metrics(ti_raw[te], ti_s[te]), noise_subtracted=metrics(ti_ns[te], ti_s[te]), full_correction=metrics(ti_full[te], ti_s[te]))
    out["mean_attenuation_variance_ratio_by_U"] = {f"{lo}-{hi}": float(np.median(ratio[(U_l >= lo) & (U_l < hi)])) for lo, hi in bins}
    out["gust_factor_raw_lidar_vs_sonic"] = dict(bias=float((gf_l - gf_s).mean()), corr=float(np.corrcoef(gf_l, gf_s)[0, 1]), mean_sonic=float(gf_s.mean()), mean_lidar=float(gf_l.mean()))
    out["R_poly"] = c.tolist()
    json.dump(out, open(os.path.join(a.out, "lidar_signal_processing.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k not in ("T_emp", "freq", "R_poly")}, indent=1))
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    for (lo, hi), col in zip(bins, ("C0", "C1", "C2")):
        T = np.array(T_emp[f"{lo}-{hi}"]); Um = U[(U >= lo) & (U < hi)].mean()
        ax[0].semilogy(fl[1:], T[1:], "o-", c=col, label=f"empirical {lo}-{hi} m/s")
        L = out["best_Leff_boxcar_m"][f"{lo}-{hi}"]; ax[0].semilogy(fl[1:], np.sinc(fl[1:] * L / Um) ** 2 * np.sinc(fl[1:] * TS) ** 2, "--", c=col, lw=1)
    ax[0].set_xlabel("f [Hz]"); ax[0].set_ylabel("T(f) = S_lidar / S_sonic"); ax[0].legend(fontsize=7); ax[0].set_title("lidar spectral transfer (dashed: best boxcar fit)")
    ax[1].scatter(ti_s[te], ti_raw[te], s=6, label="raw"); ax[1].scatter(ti_s[te], ti_full[te], s=6, label="corrected"); ax[1].plot([.05, .22], [.05, .22], "k--")
    ax[1].set_xlabel("true TI (sonic point)"); ax[1].set_ylabel("lidar TI"); ax[1].legend(); ax[1].set_title("turbulence intensity restoration (test half)")
    ax[2].hist([sn2_hat / sig_n ** 2], bins=40, range=(0, 6)); ax[2].axvline(1, c="k", ls="--"); ax[2].set_xlabel("estimated / true noise variance"); ax[2].set_title("blind noise-floor estimate")
    fig.tight_layout(); fig.savefig(os.path.join(a.out, "signal_processing.png"), dpi=130)
