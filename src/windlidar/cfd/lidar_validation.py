"""Connect the OpenFOAM RANS case to the lidar part of the project.

I wanted the CFD to be more than a stand-alone verification, so I use it in two ways.
  validation: take median FESSTVaL lidar wind profiles for three regimes, fit u* and z0 on the 66-146 m gates (the 13 and 40 m gates have no
              valid data), run the OpenFOAM case with those values and compare it with the measured profile at 150-437 m, which was not fitted.
              Also: how well does a log law (= neutral RANS) describe real 10-min profiles, by hour of day?
  closure:    use the RANS sweep (k, U at 90 m) to get sigma_u, correct it to the observed neutral level, derive the implied IEC turbulence
              class per roughness, and ask what the virtual lidar would measure at those conditions.
Limits: medians instead of single profiles, high wind is assumed near-neutral without an independent stability measurement, one site.
Usage: python -m windlidar.cfd.lidar_validation [validation|closure|all]   (needs OpenFOAM in the environment)
"""
import argparse, glob, json, os, shutil, subprocess, sys
from windlidar.paths import rp
import numpy as np, pandas as pd, xarray as xr
from windlidar.paths import ROOT
OUT = os.path.join(ROOT, "results", "analysis"); os.makedirs(OUT, exist_ok=True)
KAPPA = 0.41


def load_profiles(zmax=450.0):
    rows = []
    for f in sorted(glob.glob(os.path.join(ROOT, "real_data/falkenberg_dlidcsm/level2dwd/*/*.nc"))):
        d = xr.open_dataset(f); h = d.height.values; j = np.where(h <= zmax)[0]
        W = d.wspeed.values[:, j]; Q = d.qwind.values[:, j]
        for i, t in enumerate(d.time.values):
            rows.append((t, W[i], Q[i]))
    z = h[j]
    t = pd.DatetimeIndex([r[0] for r in rows]); W = np.array([r[1] for r in rows]); Q = np.array([r[2] for r in rows])
    return z, t, W, Q


def loglaw_fit(z, U):
    a, b = np.polyfit(np.log(z), U, 1); return KAPPA * a, float(np.exp(-b / a)), float(np.sqrt(np.mean((np.polyval([a, b], np.log(z)) - U) ** 2)))


def run_rans(name, ustar, z0):
    case = os.path.join(ROOT, "outputs", "validation", name)
    shutil.rmtree(case, ignore_errors=True); os.makedirs(case)
    for d in ("system", "constant"): shutil.copytree(os.path.join(ROOT, "openfoam", d), os.path.join(case, d))
    subprocess.run([sys.executable, "-m", "windlidar.cfd.case_builder", "--case", case, "--ustar", f"{ustar:.5f}", "--z0", f"{z0:.5f}", "--nz", "80"], check=True, capture_output=True)
    subprocess.run("blockMesh > log.blockMesh 2>&1 && simpleFoam > log.solver 2>&1", shell=True, cwd=case, check=True)
    subprocess.run([sys.executable, "-m", "windlidar.cfd.postprocess", "--case", case], check=True, capture_output=True)
    return np.loadtxt(os.path.join(case, "profile.csv"), delimiter=",", skiprows=1)


def validate_against_lidar():
    z, t, W, Q = load_profiles()
    fit_g = (z >= 60) & (z <= 150); ok_fit = np.isfinite(W[:, fit_g]).all(1) & (Q[:, fit_g] == 1).all(1)
    hour = t.hour.values; U90 = W[:, np.argmin(abs(z - 92.7))]          # gate closest to the 90 m sonic
    # per-record log-law adequacy by hour
    rms = np.full(len(t), np.nan); us = np.full(len(t), np.nan); z0s = np.full(len(t), np.nan)
    for i in np.where(ok_fit & (U90 > 4))[0]:
        us[i], z0s[i], rms[i] = loglaw_fit(z[fit_g], W[i, fit_g])
    df = pd.DataFrame(dict(hour=hour, U90=U90, rms=rms, ustar=us, z0=z0s)).dropna()
    by_hour = df.groupby("hour").agg(n=("rms", "size"), rms=("rms", "median"), U90=("U90", "median"), z0_median=("z0", "median"), ustar_median=("ustar", "median"))
    by_hour["rms_rel_pct"] = 100 * by_hour.rms / by_hour.U90
    groups = {"high wind (U92 9-13 m/s), any hour": (U90 >= 9) & (U90 < 13),
              "daytime moderate (10-15 UTC, U 4-8)": (hour >= 10) & (hour <= 15) & (U90 >= 4) & (U90 < 8),
              "night moderate (22-02 UTC, U 4-8)": ((hour >= 22) | (hour <= 2)) & (U90 >= 4) & (U90 < 8)}
    res = {"loglaw_misfit_by_hour": by_hour.round(3).reset_index().to_dict("records"), "cases": {}}
    for k, (name, m) in enumerate(groups.items()):
        sel = m & ok_fit
        Wm = np.where(Q[sel] == 1, W[sel], np.nan); n = int(sel.sum())
        M = np.nanmedian(Wm, axis=0); cnt = np.isfinite(Wm).sum(0)
        M = np.where(cnt >= 0.3 * n, M, np.nan)          # keep gates with >= 30 % valid data
        ustar, z0, rms_fit = loglaw_fit(z[fit_g], M[fit_g]); z0c = float(np.clip(z0, 0.005, 0.2)); clipped = z0c != z0
        if clipped:      # fitted z0 is outside what the mesh/wall function supports AND physically implausible (>0.2 m) -> neutral log law not applicable
            res["cases"][name] = dict(n_profiles=n, ustar_fit=float(ustar), z0_fit=float(z0), applicable=False,
                                      note="log-law fit gives an unphysical roughness; neutral RANS cannot represent this profile (no CFD comparison made)")
            continue
        P = run_rans(f"case{k}", ustar, z0c); zc, Uc = P[:, 0], P[:, 1]
        Ucfd = np.interp(z, zc, Uc)
        e_fit = (Ucfd - M)[fit_g]; far = (z > 150) & np.isfinite(M)
        res["cases"][name] = dict(n_profiles=n, applicable=True, ustar_fit=float(ustar), z0_fit=float(z0), z0_used=z0c, loglaw_fit_rms=float(rms_fit),
                                  cfd_vs_measured_rms_fit_66_146=float(np.sqrt(np.mean(e_fit ** 2))), cfd_vs_measured_bias_150_450=float(np.nanmean((Ucfd - M)[far])),
                                  cfd_vs_measured_rms_150_450=float(np.sqrt(np.nanmean((Ucfd - M)[far] ** 2))), highest_valid_gate_m=float(z[far].max()), n_far_gates=int(far.sum()))
    return res


def vl_ti(args):
    from windlidar.lidar import virtual as V
    U, sigma, seed = args; rng = np.random.default_rng(seed); nv, pts, ns = V.geometry()
    F = V.field(U, sigma, 340.0, pts, rng); vr = V.scan(F, U, nv, ns, sigma_r=0.2, rng=rng); u, v, w = V.retrieve(vr, nv)
    from scipy.signal import welch
    s = np.hypot(u, v); ss = V.sonic_gust(F, U)
    fl, Pl = welch(s, fs=1 / 3.4, nperseg=32, noverlap=16, detrend="linear"); sn2 = np.median(Pl[fl > 0.115]) / (2 * 3.4)   # blind noise floor, same as signal_processing.py
    return ss["ti"], s.std() / s.mean(), s.mean(), s.std(), sn2


def closure_check():
    from multiprocessing import Pool
    from windlidar.design.iec import IREF, ntm_sigma1
    d = pd.read_csv(os.path.join(ROOT, "data", "rans_sweep.csv"))
    d["sigma_rans"] = np.sqrt(d.k90 / 0.945); d["ratio_rans"] = d.sigma_rans / d.ustar
    d["k_over_ustar2_sqrtCmu"] = d.k90 / (d.ustar ** 2 / np.sqrt(0.09))
    R_obs = 12 ** (1 / 3)                       # neutral sigma_u/u* = 2.29 (Panofsky-Dutton limit; observed typical 2.4)
    d["corr_factor"] = R_obs * d.ustar / d.sigma_rans; d["sigma_corr"] = R_obs * d.ustar
    d["Iref_implied"] = d.sigma_corr / (0.75 * d.U90 + 5.6)
    d["IEC_class_mean_state"] = [("A+" if i >= .18 else "A" if i >= .16 else "B" if i >= .14 else "C" if i >= .12 else "<C") for i in d.Iref_implied]
    sp = json.load(open(os.path.join(OUT, "lidar_signal_processing.json"))); c = np.array(sp["R_poly"])
    jobs = [(r.U90, r.sigma_corr, 100 * i + j) for i, r in d.iterrows() for j in range(12)]
    with Pool(16) as p: out = np.array(p.map(vl_ti, jobs)).reshape(len(d), 12, 5)
    d["TI_true"] = out[:, :, 0].mean(1); d["TI_lidar_raw"] = out[:, :, 1].mean(1)
    var_ns = np.clip(out[:, :, 3] ** 2 - out[:, :, 4], 1e-4, None)      # identical pipeline to signal_processing.py: blind noise subtraction, then R(U)
    d["TI_lidar_restored"] = (np.sqrt(var_ns / np.clip(np.polyval(c, out[:, :, 2]), .1, 1.5)) / out[:, :, 2]).mean(1)
    d["TI_if_uncorrected_rans"] = d.sigma_rans / d.U90
    cols = ["ustar", "z0", "U90", "sigma_rans", "sigma_corr", "TI_if_uncorrected_rans", "TI_true", "TI_lidar_raw", "TI_lidar_restored", "Iref_implied", "IEC_class_mean_state"]
    d[cols].to_csv(os.path.join(OUT, "cfd_design_table.csv"), index=False)
    return d[cols], dict(rans_sigma_over_ustar=float(d.ratio_rans.mean()), observed_neutral_sigma_over_ustar=float(R_obs), mean_correction_factor=float(d.corr_factor.mean()),
                         k_over_expected_min_max=[float(d.k_over_ustar2_sqrtCmu.min()), float(d.k_over_ustar2_sqrtCmu.max())])


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("part", choices=["validation", "closure", "all"], default="all", nargs="?"); a = ap.parse_args(); res = {}
    if a.part in ("validation", "all"): res["validation"] = validate_against_lidar(); print(json.dumps(res["validation"]["cases"], indent=1))
    if a.part in ("closure", "all"):
        t, info = closure_check(); res["closure"] = info; print(info); print(t.round(3).to_string())
    for key, fname in (("validation", "cfd_lidar_validation.json"), ("closure", "cfd_closure_check.json")):
        if key in res: json.dump({key: res[key]}, open(os.path.join(OUT, fname), "w"), indent=1)
