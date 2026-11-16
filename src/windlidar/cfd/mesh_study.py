"""Richardson extrapolation and grid-convergence index (Celik et al. 2008) over three meshes (nz = 40/80/160, refinement ratio 2).

Why: a CFD result without a discretisation-error estimate is not verified. I report the GCI for U(90 m), k(90 m) and the fitted u*.
"""
import argparse, glob, json, os
import numpy as np

def gci(f_fine, f_med, f_coarse, r=2.0):
    e21, e32 = f_med - f_fine, f_coarse - f_med
    if abs(e21) < 1e-14 or e32 / e21 <= 0:
        return dict(p=None, note="non-monotone or converged exactly; GCI not defined", f_ext=None, gci_fine_pct=None)
    p = np.log(abs(e32 / e21)) / np.log(r)
    f_ext = f_fine + (f_fine - f_med) / (r ** p - 1)
    gci_f = 1.25 * abs((f_fine - f_med) / f_fine) / (r ** p - 1) * 100
    return dict(p=float(p), f_ext=float(f_ext), gci_fine_pct=float(gci_f), note="")

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--runs", default="runs"); a = ap.parse_args()
    runs = sorted(glob.glob(os.path.join(a.runs, "nz*", "metrics.json")), key=lambda p: int(os.path.basename(os.path.dirname(p))[2:]))
    if len(runs) < 3:
        raise SystemExit("need three runs, found %d" % len(runs))
    M = [json.load(open(p)) for p in runs[:3]]          # coarse, medium, fine
    out = {}
    print("%6s %10s %10s %12s %12s" % ("nz", "U90[m/s]", "u*_fit", "U err mean%", "U err max%"))
    for m in M:
        print("%6d %10.4f %10.4f %12.3f %12.3f" % (m["nz"], m["U90"], m["ustar_fit"], m["U_err_mean_pct"], m["U_err_max_pct"]))
    for key in ("U90", "ustar_fit", "k90"):
        out[key] = gci(M[2][key], M[1][key], M[0][key], r=M[2]["nz"] / M[1]["nz"])
        print(key, out[key])
    json.dump(out, open(os.path.join(a.runs, "gci.json"), "w"), indent=1)

if __name__ == "__main__":
    main()
