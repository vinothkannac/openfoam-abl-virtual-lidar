"""Extract the vertical profile from the last OpenFOAM time step and compare it with the Richards-Hoxey solution.

I read the ASCII fields directly (no ParaView). Writes profile.csv, metrics.json and profile.png into the case directory.
The fitted u* is taken between --zmin and --zmax (default 10-150 m) because the lowest cells are dominated by the wall function.
"""
import argparse, json, os
from windlidar.paths import rp
import numpy as np
from windlidar.cfd.foam_io import read_internal_field, latest_time_dir
from windlidar.cfd.case_builder import rh_profiles, KAPPA

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default=".")
    ap.add_argument("--zmin", type=float, default=10.0)
    ap.add_argument("--zmax", type=float, default=150.0)
    a = ap.parse_args()
    P = json.load(open(os.path.join(a.case, "case_params.json")))
    nx, nz, zc = P["nx"], P["nz"], np.array(P["zc"])
    t = latest_time_dir(a.case)
    U = read_internal_field(os.path.join(a.case, t, "U"))
    k = read_internal_field(os.path.join(a.case, t, "k"))
    eps = read_internal_field(os.path.join(a.case, t, "epsilon"))
    ix = nx // 2                                   # column in the middle of the domain
    idx = ix + nx * np.arange(nz)                  # blockMesh cell order: x fastest, then y, then z
    Ux, kc, ec = U[idx, 0], k[idx], eps[idx]
    Ur, kr, er = rh_profiles(zc, P["ustar"], P["z0"])

    sel = (zc >= a.zmin) & (zc <= a.zmax)
    relU = (Ux - Ur) / Ur
    # friction velocity from a log-law fit U = (u*/kappa) ln(z+z0) - (u*/kappa) ln z0
    slope = np.polyfit(np.log(zc[sel] + P["z0"]), Ux[sel], 1)[0]
    ustar_fit = slope * KAPPA
    U90 = float(np.interp(90.0, zc, Ux)); U90_ref = float(np.interp(90.0, zc, Ur))
    k90 = float(np.interp(90.0, zc, kc)); e90 = float(np.interp(90.0, zc, ec))
    m = dict(t=t, nz=nz, ustar=P["ustar"], z0=P["z0"], U_err_mean_pct=float(100 * np.mean(np.abs(relU[sel]))),
             U_err_max_pct=float(100 * np.max(np.abs(relU[sel]))), ustar_fit=float(ustar_fit),
             ustar_err_pct=float(100 * (ustar_fit - P["ustar"]) / P["ustar"]),
             U90=U90, U90_ref=U90_ref, U90_err_pct=float(100 * (U90 - U90_ref) / U90_ref),
             k90=k90, eps90=e90,
             sigma_u_rans=float(np.sqrt(k90 / 0.945)),   # assumes sigma_u:sigma_v:sigma_w = 1:0.8:0.5
             sigma_u_obs_typ=float(2.4 * P["ustar"]))
    json.dump(m, open(os.path.join(a.case, "metrics.json"), "w"), indent=1)
    np.savetxt(os.path.join(a.case, "profile.csv"), np.c_[zc, Ux, Ur, kc, kr, ec, er], delimiter=",",
               header="z,U_cfd,U_ref,k_cfd,k_ref,eps_cfd,eps_ref", comments="")
    print(json.dumps(m, indent=1))
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 3, figsize=(11, 4))
        for a_, c, r, lab in zip(ax, (Ux, kc, ec), (Ur, kr, er), ("U [m/s]", "k [m2/s2]", "epsilon [m2/s3]")):
            a_.plot(c, zc, "o-", ms=3, label="OpenFOAM"); a_.plot(r, zc, "--", label="Richards-Hoxey")
            a_.set_xlabel(lab); a_.set_ylim(0, 200); a_.grid(alpha=.3)
        ax[0].set_ylabel("z [m]"); ax[0].legend()
        fig.tight_layout(); fig.savefig(os.path.join(a.case, "profile.png"), dpi=130)
    except Exception as e:
        print("plot skipped:", e)

if __name__ == "__main__":
    main()
