"""Regenerate every figure in docs/figures from the stored results (no heavy recomputation).

    python -m windlidar.figures
"""
import json, os
import numpy as np, pandas as pd
from windlidar import viz
from windlidar.viz import plt, C, INK2, MUTED
from windlidar.paths import rp

FIG = rp("docs", "figures"); os.makedirs(FIG, exist_ok=True)
J = lambda *p: json.load(open(rp(*p)))


def fig_cfd_profile():
    d = np.genfromtxt(rp("outputs/mesh_study/nz80/profile.csv"), delimiter=",", names=True)
    gci = J("outputs/mesh_study/gci.json")
    fig, ax = plt.subplots(1, 3, figsize=(14, 4.4), gridspec_kw=dict(width_ratios=[1, 1, 1.1], wspace=.45))
    ax[0].plot(d["U_ref"], d["z"], c=C["orange"], lw=1.6, ls="--", label="Richards–Hoxey (analytic)")
    ax[0].plot(d["U_cfd"], d["z"], c=C["blue"], label="OpenFOAM RANS, k–ε")
    ax[0].set(yscale="log", ylim=(0.2, 1000), xlabel="U  [m/s]", ylabel="height z  [m]", title="Mean wind"); ax[0].legend(loc="upper left")
    ax[1].plot(d["k_ref"], d["z"], c=C["orange"], lw=1.6, ls="--"); ax[1].plot(d["k_cfd"], d["z"], c=C["blue"])
    ax[1].set(yscale="log", ylim=(0.2, 1000), xlabel=r"k  [m$^2$/s$^2$]", title="Turbulent kinetic energy"); ax[1].set_xlim(0.78, 0.9); ax[1].axvline(0.8333, c=MUTED, lw=.8, ls=":")
    names = {"U90": "U(90 m)", "k90": "k(90 m)", "ustar_fit": "u* (fit)"}
    vals = [abs(gci[k]["gci_fine_pct"]) for k in names]
    ax[2].barh(list(names.values()), vals, color=C["blue"], height=.5)
    for i, v in enumerate(vals): ax[2].text(v + .05, i, f"{v:.2f} %", va="center", color=INK2)
    ax[2].set(xlabel="GCI, fine grid  [%]", title="Mesh convergence (nz = 40 / 80 / 160)"); ax[2].grid(axis="y", visible=False)
    ax[2].set_xlim(0, max(vals) * 1.25)
    viz.title(fig, "Neutral ABL with OpenFOAM: the RANS solution reproduces the analytic profile",
              "simpleFoam, k–ε, u* = 0.5 m/s, z0 = 0.05 m, 80 cells over 1 km")
    viz.save(fig, f"{FIG}/cfd_abl_profile.png")


def fig_cfd_sweep():
    d = pd.read_csv(rp("data/rans_sweep.csv"))
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 4))
    for a, col, t, cm in ((ax[0], "U90", "U at 90 m  [m/s]", "Blues"), (ax[1], "k90", "k at 90 m  [m²/s²]", "Oranges")):
        p = d.pivot(index="z0", columns="ustar", values=col)
        im = a.imshow(p.values, origin="lower", cmap=cm, aspect="auto"); a.grid(False)
        a.set_xticks(range(p.shape[1]), [f"{x:g}" for x in p.columns]); a.set_yticks(range(p.shape[0]), [f"{x:g}" for x in p.index])
        a.set(xlabel="u*  [m/s]", ylabel="z0  [m]", title=t)
        for i in range(p.shape[0]):
            for j in range(p.shape[1]):
                v = p.values[i, j]; a.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8.5, color="white" if v > p.values.mean() else INK2)
        a.spines[:].set_visible(False)
    viz.title(fig, "RANS parameter sweep (16 runs)", "k depends on u* only (inlet BCs fix it), so the sweep adds no independent turbulence information")
    viz.save(fig, f"{FIG}/cfd_sweep.png")


def fig_gustnet():
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.2), gridspec_kw=dict(width_ratios=[1.2, 1]))
    for (n, col) in zip(("A_small_relu", "B_medium_relu", "C_medium_gelu", "D_deep_relu"), (C["blue"], C["orange"], C["aqua"], C["violet"])):
        h = J(f"models/baseline/hist_{n}.json"); ax[0].plot(h["val"], c=col, label=n.replace("_", " "))
    ax[0].set(yscale="log", xlabel="epoch", ylabel="validation loss (MSE)", title="Training: four MLP configurations"); ax[0].legend()
    m = J("results/gustnet/metrics_level2.json")
    rows = [("raw lidar", m["baseline_raw_lidar"]), ("linear fit", m["baseline_linear"]), ("MLP (best)", m["models"]["B_medium_relu"])]
    x = np.arange(3); w = .36
    for k, (key, col, lab) in enumerate((("test", C["blue"], "held-out simulations"), ("unseen", C["orange"], "unseen parameter ranges"))):
        v = [r[1][key]["R2"] for r in rows]; ax[1].bar(x + (k - .5) * w, v, w * .92, color=col, label=lab)
        for xi, vi in zip(x + (k - .5) * w, v): ax[1].text(xi, vi + .015, f"{vi:.2f}", ha="center", fontsize=8.5, color=INK2)
    ax[1].set_xticks(x, [r[0] for r in rows]); ax[1].set(ylabel="R²  (sonic gust factor)", ylim=(0, 1), title="Synthetic benchmark"); ax[1].legend(loc="upper left"); ax[1].grid(axis="x", visible=False)
    viz.title(fig, "Gust-factor correction network on synthetic (Kaimal) data", "inputs: lidar mean wind and lidar gust factor; target: sonic gust factor")
    viz.save(fig, f"{FIG}/gustnet_synthetic.png")


def fig_fesstval():
    R = J("results/fesstval/real_analysis.json")
    order = [("raw", "raw lidar"), ("synthetic_nn", "network\n(synthetic\ntraining)"), ("lin_gf", "linear\nrecalib."), ("lin_diurnal", "+ hour of day"), ("gbr_diurnal", "gradient\nboosting")]
    fig, ax = plt.subplots(1, 2, figsize=(13.5, 4.6), sharey=True)
    for a, p, name in zip(ax, ("dwd", "uzk"), ("DWD processing", "Cologne (UzK) processing")):
        x = np.arange(len(order)); w = .26
        for k, (sp, col, lab) in enumerate((("holdout", C["blue"], "all"), ("holdout_day", C["orange"], "day 09–17 UTC"), ("holdout_night", C["violet"], "night"))):
            v = [R[p][sp][n]["R2"] for n, _ in order]; a.bar(x + (k - 1) * w, v, w * .9, color=col, label=lab)
            if sp == "holdout":
                for xi, vi in zip(x + (k - 1) * w, v): a.text(xi, max(vi, 0) + .012, f"{vi:.2f}", ha="center", fontsize=8, color=INK2)
        a.set_xticks(x, [l for _, l in order], fontsize=8.5); a.set_title(name); a.grid(axis="x", visible=False); a.axhline(0, c=MUTED, lw=.8)
    ax[0].set_ylabel("R²  on chronological hold-out"); ax[0].set_ylim(-1.0, 0.6); ax[0].legend(loc="lower left", ncol=3)
    viz.title(fig, "Real data (FESSTVaL 2021): a lidar-to-sonic gust correction is only partly learnable",
              "Falkenberg, 90 m; 10-min gust factor; models trained on the first part of the campaign, tested afterwards")
    viz.save(fig, f"{FIG}/fesstval_model_ladder.png")

    fig, ax = plt.subplots(1, 3, figsize=(13, 4))
    for a, p, name in zip(ax[:2], ("dwd", "uzk"), ("DWD", "UzK")):
        M = pd.read_csv(rp(f"results/fesstval/matched_{p}.csv"), parse_dates=["t"])
        g = M.groupby("hour")[["gf_s", "gf_l"]].mean(); a.plot(g.index, g.gf_s, c=C["blue"], label="sonic"); a.plot(g.index, g.gf_l, c=C["orange"], label=f"lidar ({name})")
        a.axvspan(19, 24, color=C["violet"], alpha=.08, lw=0); a.axvspan(0, 5, color=C["violet"], alpha=.08, lw=0)
        a.set(xlabel="hour of day (UTC)", ylabel="mean 10-min gust factor", xlim=(0, 23), title=f"Diurnal cycle: {name}"); a.legend()
    M = pd.read_csv(rp("results/fesstval/matched_dwd.csv"))
    ax[2].hexbin(M.gf_s, M.gf_l, gridsize=40, cmap="Blues", mincnt=1, bins="log"); lim = (1, 2.2); ax[2].plot(lim, lim, c=INK2, ls="--", lw=1)
    ax[2].set(xlim=lim, ylim=lim, xlabel="sonic gust factor", ylabel="lidar gust factor (DWD)", title="Joint distribution"); ax[2].grid(False)
    viz.title(fig, "Lidar and sonic gust factors differ systematically", "shaded: stable-night hours")
    viz.save(fig, f"{FIG}/fesstval_diurnal.png")


def fig_stability():
    S = J("results/stability_ablation.json")
    rows = []
    for k, v in S.items():
        e, f, p = k.split("|"); rows.append((e, f, p, v["holdout"]["R2"]))
    D = pd.DataFrame(rows, columns=["emu", "feat", "prod", "R2"]); D = D[D.feat == "level2_hour"]
    lab = {"full": "full stability emulator", "no_decoherence": "no decoherence", "neutral_only": "neutral only", "neutral_no_decoh": "neutral, no decoherence", "const_noise": "constant noise"}
    fig, ax = plt.subplots(figsize=(9.5, 4)); idx = D.emu.unique(); x = np.arange(len(idx)); w = .36
    for k, (p, col) in enumerate((("dwd", C["blue"]), ("uzk", C["orange"]))):
        v = [D[(D.emu == e) & (D.prod == p)].R2.mean() for e in idx]; ax.bar(x + (k - .5) * w, v, w * .9, color=col, label=p.upper())
        for xi, vi in zip(x + (k - .5) * w, v): ax.text(xi, vi + .01, f"{vi:.2f}", ha="center", fontsize=8.5, color=INK2)
    ax.set_xticks(x, [lab.get(e, e) for e in idx], fontsize=8.5); ax.set(ylabel="hold-out R² on real data"); ax.legend(); ax.grid(axis="x", visible=False)
    viz.title(fig, "Stability-aware synthetic training does not close the sim-to-real gap", "network inputs: lidar U, lidar GF, hour of day")
    viz.save(fig, f"{FIG}/stability_ablation.png")


def fig_lidar_sp():
    sp = J("results/analysis/lidar_signal_processing.json"); ns = J("results/analysis/noise_sweep.json")
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.4), gridspec_kw=dict(wspace=.4))
    f = np.array(sp["freq"])
    for (k, T), col in zip(sp["T_emp"].items(), (C["blue"], C["orange"], C["aqua"])):
        ax[0].semilogy(f[1:], np.array(T)[1:], "o-", c=col, ms=3.5, label=f"{k} m/s  (L_eff = {sp['best_Leff_boxcar_m'][k]:.0f} m)")
    ax[0].set(xlabel="frequency  [Hz]", ylabel="T(f) = S_lidar / S_sonic", title="Spectral transfer function of the CSM scan"); ax[0].legend()
    ti = sp["TI_test"]; names = {"raw": "raw", "noise_subtracted": "+ blind noise\nsubtraction", "full_correction": "+ attenuation\ncorrection"}
    x = np.arange(3); ax[1].bar(x, [ti[k]["bias"] for k in names], .55, color=[C["red"], C["yellow"], C["aqua"]]); ax[1].set_ylim(-.04, .004)
    for xi, k in zip(x, names): ax[1].text(xi, ti[k]["bias"] - .0035, f"{ti[k]['bias']:+.3f}", ha="center", fontsize=9, color=INK2)
    ax[1].axhline(0, c=MUTED, lw=.8); ax[1].set_xticks(x, list(names.values()), fontsize=9); ax[1].set(ylabel="TI bias vs sonic", title="Turbulence-intensity restoration"); ax[1].grid(axis="x", visible=False)
    s = sorted(ns, key=float); ax[2].plot([float(k) for k in s], [ns[k]["gf_bias"] for k in s], "o-", c=C["blue"])
    ax[2].axhline(0.04, c=C["orange"], ls="--", lw=1.4); ax[2].text(0.12, 0.043, "real lidar − sonic bias (+0.04)", color=INK2, fontsize=9)
    ax[2].set(xlabel="radial-velocity noise  [m/s]", ylabel="gust-factor bias (lidar − sonic)", title="Noise needed to reproduce the real bias")
    viz.title(fig, "Virtual lidar: what the scanning geometry does to the wind signal", "62° cone, 10 beams per 3.4 s, 30 m gates; Veers-synthesised 3-D field with IEC coherence")
    viz.save(fig, f"{FIG}/lidar_signal_processing.png")


def fig_cfd_validation():
    A = J("results/analysis/cfd_lidar_validation.json")["validation"]; H = pd.DataFrame(A["loglaw_misfit_by_hour"])
    fig, ax = plt.subplots(1, 3, figsize=(14.5, 4.2), gridspec_kw=dict(width_ratios=[1.25, 1, 1]))
    ax[0].semilogy(H.hour, H.z0_median, "o-", c=C["blue"]); ax[0].axvspan(-1, 4.5, color=C["violet"], alpha=.08, lw=0); ax[0].axvspan(19, 23, color=C["violet"], alpha=.08, lw=0)
    ax[0].axhline(0.1, c=MUTED, ls=":", lw=1); ax[0].text(9, 0.14, "physical range for farmland  z0 ≲ 0.1 m", color=INK2, fontsize=8.5)
    ax[0].set(xlabel="hour (UTC)", ylabel="fitted roughness z0  [m]", title="Log-law fit to lidar profiles"); ax[0].text(.2, 20, "stable night:\nlog-law invalid", color=C["violet"], fontsize=9)
    c = A["cases"]; names = [k.split("(")[0].strip() for k in c]; ap = [v.get("applicable") for v in c.values()]
    ok = [(k, v) for k, v in c.items() if v.get("applicable")]
    y = np.arange(len(ok))
    ax[1].barh(y - .18, [v["cfd_vs_measured_rms_150_450"] for _, v in ok], .34, color=C["blue"], label="RMS"); ax[1].barh(y + .18, [abs(v["cfd_vs_measured_bias_150_450"]) for _, v in ok], .34, color=C["orange"], label="|bias|")
    ax[1].set_yticks(y, [k.split("(")[0].strip() for k, _ in ok], fontsize=9); ax[1].invert_yaxis(); ax[1].set(xlabel="m/s", title="CFD vs lidar, 150–437 m (not fitted)"); ax[1].legend(); ax[1].grid(axis="y", visible=False)
    ax[2].plot(H.hour, H.rms_rel_pct, "o-", c=C["orange"]); ax[2].set(xlabel="hour (UTC)", ylabel="log-law RMS misfit [% of U]", title="Misfit of the neutral log law"); ax[2].set_ylim(0, None)
    viz.title(fig, "Where the neutral RANS model is valid: real lidar profiles (FESSTVaL 2021)", "u*, z0 fitted to gates 66–146 m; the 150–437 m comparison is independent. Not applicable on stable nights.")
    viz.save(fig, f"{FIG}/cfd_validation.png")


def fig_design():
    I = J("results/analysis/iec_design_check.json"); N = pd.DataFrame(I["NTM_table"])
    fig, ax = plt.subplots(1, 3, figsize=(15.5, 4.5), gridspec_kw=dict(width_ratios=[1.3, 1, .9], wspace=.35))
    ax[0].plot(N.U, N.raw_p90, "o-", c=C["red"], label="measured p90 (raw lidar)"); ax[0].plot(N.U, N.corr_p90, "o-", c=C["blue"], label="measured p90 (corrected)")
    for k, col in (("NTM_A", C["violet"]), ("NTM_B", C["aqua"]), ("NTM_C", C["yellow"])): ax[0].plot(N.U, N[k], ls="--", c=col, lw=1.4, label=f"IEC NTM class {k[-1]}")
    ax[0].set(xlabel="mean wind at 92 m  [m/s]", ylabel="σu  [m/s]", title="Measured σu against IEC NTM"); ax[0].legend(fontsize=8, loc="upper left")
    e = J("results/analysis/iec_design_check.json")
    D = pd.read_csv(rp("results/analysis/cfd_design_table.csv"))
    ax[1].scatter(D.ustar + np.log10(D.z0) * 0.0, D.Iref_implied, c=np.log10(D.z0), cmap="Blues_r", s=60, edgecolor=viz.INK2, linewidth=.6)
    for lvl, nm in ((.12, "C"), (.14, "B"), (.16, "A")): ax[1].axhline(lvl, c=MUTED, ls=":", lw=1); ax[1].text(.31, lvl + .002, f"class {nm}", fontsize=8.5, color=INK2)
    ax[1].set(xlabel="u*  [m/s]", ylabel="implied Iref (mean state)", ylim=(0.05, .17), title="CFD-implied turbulence class")
    ax[2].bar(["raw lidar", "restored"], [D.TI_lidar_raw.div(D.TI_true).mean() - 1, D.TI_lidar_restored.div(D.TI_true).mean() - 1], .5, color=[C["red"], C["aqua"]])
    for i, v in enumerate([D.TI_lidar_raw.div(D.TI_true).mean() - 1, D.TI_lidar_restored.div(D.TI_true).mean() - 1]): ax[2].text(i, v + (.006 if v > 0 else -.02), f"{v:+.1%}", ha="center", color=INK2)
    ax[2].axhline(0, c=MUTED, lw=.8); ax[2].set_ylim(-.22, .06); ax[2].set(ylabel="TI error of virtual lidar", title="Lidar TI at CFD design points"); ax[2].grid(axis="x", visible=False)
    viz.title(fig, "Design wind conditions: measurement, closure and IEC 61400-1 classes", "NTM σ1 = Iref (0.75 U + 5.6); corrected lidar σu; CFD closure σu = 1.88 u* × 1.22")
    viz.save(fig, f"{FIG}/design_conditions.png")


def fig_ddpm():
    E = J("results/analysis/diffusion_eval.json"); keys = ["normal", "EOG_A=3", "EOG_A=5"]
    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    bands = ["0.05-0.2Hz", "0.2-1Hz", "1-5Hz"]; x = np.arange(3); w = .26
    for k, (kk, col) in enumerate(zip(keys, (C["blue"], C["orange"], C["violet"]))):
        ax[0].bar(x + (k - 1) * w, [E[kk]["psd_ratio_gen_over_ref"][b] for b in bands], w * .9, color=col, label=kk)
    ax[0].axhline(1, c=INK2, ls="--", lw=1); ax[0].set_xticks(x, bands); ax[0].set(ylim=(0, 1.2), ylabel="PSD ratio  generated / reference", title="Spectral fidelity"); ax[0].legend(loc="lower right"); ax[0].grid(axis="x", visible=False)
    ax[1].scatter([E[k]["increment_kurtosis_1s"]["ref"] for k in keys], [E[k]["increment_kurtosis_1s"]["gen"] for k in keys], c=[C["blue"], C["orange"], C["violet"]], s=70)
    ax[1].plot([2.5, 9], [2.5, 9], c=INK2, ls="--", lw=1); ax[1].set(xlabel="reference", ylabel="diffusion model", title="Kurtosis of 1 s increments (intermittency)")
    g = [k for k in keys if "gust_amplitude_estimate" in E[k]]
    ax[2].errorbar([E[k]["gust_amplitude_estimate"]["requested"] for k in g], [E[k]["gust_amplitude_estimate"]["gen_mean"] for k in g], [E[k]["gust_amplitude_estimate"]["gen_sd"] for k in g], fmt="o", c=C["blue"], capsize=3, label="diffusion model")
    ax[2].plot([2.5, 5.5], [2.5, 5.5], c=INK2, ls="--", lw=1, label="ideal"); ax[2].set(xlabel="requested EOG amplitude  [m/s]", ylabel="generated amplitude  [m/s]", title="Gust conditioning"); ax[2].legend()
    viz.title(fig, "Conditional diffusion model for wind series", "1-D U-Net DDPM, 200 steps; conditioned on mean wind, turbulence intensity and gust amplitude")
    viz.save(fig, f"{FIG}/diffusion_wind.png")


if __name__ == "__main__":
    for f in (fig_cfd_profile, fig_cfd_sweep, fig_gustnet, fig_fesstval, fig_stability, fig_lidar_sp, fig_cfd_validation, fig_design, fig_ddpm):
        try: f()
        except Exception as ex: print("FAILED", f.__name__, repr(ex))
