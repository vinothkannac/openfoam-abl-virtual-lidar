"""IEC 61400-1 design wind conditions and a check against the FESSTVaL measurements at about 90 m.

NTM  sigma1 = Iref (0.75 Vhub + b), b = 5.6 m/s.   ETM  sigma = c Iref (0.072 (Vave/c + 3)(Vhub/c - 4) + 10), c = 2 m/s, Vave = 0.2 Vref.
EOG  Ve1 = 0.8 Ve50, Vgust = min(1.35 (Ve1 - Vhub), 3.3 sigma1 / (1 + 0.1 D/Lambda1)), V(t) = V - 0.37 Vgust sin(3 pi t/T)(1 - cos(2 pi t/T)), T = 10.5 s.
EWM  Ve50 = 1.4 Vref (z/zhub)^0.11.   Classes: Vref = 50 / 42.5 / 37.5 m/s, Iref = 0.18 (A+), 0.16 (A), 0.14 (B), 0.12 (C).
I implemented these from the standard's formulas (edition 3 style) to connect lidar statistics to design conditions. The EOG check compares
point (sonic) gusts with a rotor-averaged design gust, so it is an illustration and not an IEC exceedance statistic.
"""
import json, os
from windlidar.paths import rp
import numpy as np, pandas as pd

VREF = {"I": 50.0, "II": 42.5, "III": 37.5}
IREF = {"A+": 0.18, "A": 0.16, "B": 0.14, "C": 0.12}
LAMBDA1 = 42.0


def ntm_sigma1(V, Iref): return Iref * (0.75 * V + 5.6)
def etm_sigma(V, Iref, Vref, c=2.0): return c * Iref * (0.072 * (0.2 * Vref / c + 3) * (V / c - 4) + 10)
def ewm_ve50(Vref, z=90.0, zhub=90.0): return 1.4 * Vref * (z / zhub) ** 0.11


def eog_vgust(V, Iref, Vref, D=126.0):
    Ve1 = 0.8 * ewm_ve50(Vref)
    return np.minimum(1.35 * (Ve1 - V), 3.3 * ntm_sigma1(V, Iref) / (1 + 0.1 * D / LAMBDA1))


def eog_series(V, Vgust, T=10.5, dt=0.05):
    t = np.arange(0, T + dt, dt)
    return t, V - 0.37 * Vgust * np.sin(3 * np.pi * t / T) * (1 - np.cos(2 * np.pi * t / T))


def eog_peak_rise(Vgust, win_s=3.0, T=10.5, dt=0.05):
    """Largest rise above V of the 3-s moving mean of the EOG shape (comparable to a measured 3-s gust minus the mean)."""
    t, v = eog_series(0.0, Vgust, T, dt); w = int(round(win_s / dt)); m = np.convolve(v, np.ones(w) / w, "valid")
    return float(m.max())


def check_real(matched_sonic=rp("results/fesstval/matched_dwd.csv"), matched_lidar=rp("results/fesstval/matched_uzk.csv"), out=rp("results/analysis")):
    os.makedirs(out, exist_ok=True)
    S = pd.read_csv(matched_sonic); L = pd.read_csv(matched_lidar)
    res = {"n_sonic": len(S), "n_lidar": len(L)}
    S["dgust"] = S.gust_s - S.mean_s
    # --- EOG check: how often does a measured 3-s gust excess exceed the (3-s smoothed) EOG rise of class II-B?
    for cls, (vr, ir) in {"IIB": (VREF["II"], IREF["B"]), "IIIC": (VREF["III"], IREF["C"]), "IA": (VREF["I"], IREF["A"])}.items():
        vg = eog_vgust(S.mean_s.values, ir, vr); rise = np.array([eog_peak_rise(g) for g in vg])
        res[f"EOG_{cls}"] = dict(frac_10min_records_with_gust_excess_above_EOG_rise=float((S.dgust.values > rise).mean()),
                                  median_EOG_Vgust=float(np.median(vg)), median_EOG_3s_rise=float(np.median(rise)), median_measured_gust_excess=float(S.dgust.median()),
                                  p99_measured_gust_excess=float(S.dgust.quantile(0.99)))
    # --- NTM check with lidar turbulence (UzK sd_wspeed). Raw lidar sigma is attenuated; correct with the virtual-lidar variance ratio (assumption!)
    att = json.load(open(os.path.join(out, "lidar_signal_processing.json")))["mean_attenuation_variance_ratio_by_U"]
    r_var = float(np.mean(list(att.values())))
    L = L[(L.mean_l >= 4)].copy(); L["sig_raw"] = L.sd_l; L["sig_corr"] = L.sd_l / np.sqrt(r_var)
    bins = np.array([4, 6, 8, 10, 12, 16]); L["b"] = pd.cut(L.mean_l, bins)
    g = L.groupby("b", observed=True).agg(U=("mean_l", "mean"), n=("sig_raw", "size"), raw_p90=("sig_raw", lambda x: x.quantile(.9)), corr_p90=("sig_corr", lambda x: x.quantile(.9)), corr_mean=("sig_corr", "mean"))
    g["NTM_A"] = ntm_sigma1(g.U, IREF["A"]); g["NTM_B"] = ntm_sigma1(g.U, IREF["B"]); g["NTM_C"] = ntm_sigma1(g.U, IREF["C"])
    res["NTM_table"] = g.round(3).reset_index().astype({"b": str}).to_dict("records"); res["variance_ratio_used"] = r_var
    cls = None
    for c in ("C", "B", "A", "A+"):
        if (g.corr_p90 <= ntm_sigma1(g.U, IREF[c]) * 1.0).all(): cls = c; break
    res["lowest_NTM_class_covering_corrected_p90_sigma"] = cls
    json.dump(res, open(os.path.join(out, "iec_design_check.json"), "w"), indent=1)
    return res


if __name__ == "__main__":
    r = check_real(); print(json.dumps({k: v for k, v in r.items() if k != "NTM_table"}, indent=1))
    print(pd.DataFrame(r["NTM_table"]).to_string())
