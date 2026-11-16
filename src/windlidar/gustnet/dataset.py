"""Generate the synthetic training set: one row = one 10-minute record seen by a sonic and by a lidar.

Network inputs (lidar side): U_lidar, TI_lidar, GF_lidar. Targets (sonic side): GF_sonic, TI_sonic. The latent parameters (u*, z0,
sigma ratio, L_u, L_eff, noise) are stored for analysis but never used as inputs. u* and z0 can come from the RANS sweep.
--noise-range sets the lidar retrieval-noise range; I added it after finding that the default range made the emulated lidar gust factor too low
compared with the real lidar (see docs/EXPERIMENT_LOG.md). Usage: python -m windlidar.gustnet.dataset --n 4000
"""
import argparse, json, os, time
from windlidar.paths import rp
import numpy as np, pandas as pd
from windlidar.atmosphere.kaimal import synth_series, iec_length_scale, log_wind
from windlidar.lidar.gust_proxy import sonic_stats, lidar_stats

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4000)
    ap.add_argument("--z", type=float, default=90.0, help="measurement height [m]")
    ap.add_argument("--Ts", type=float, default=3.4, help="lidar scan cycle [s] (FESSTVaL CSM: 3.4 s)")
    ap.add_argument("--gate", type=float, default=30.0, help="range gate [m] (Streamline 30 m, XR 48 m)")
    ap.add_argument("--leff-range", type=float, nargs=2, default=[40.0, 150.0])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--noise-range", type=float, nargs=2, default=[0.05, 0.30], help="lidar retrieval noise std range [m/s]")
    ap.add_argument("--rans-csv", default=None, help="optional: use U, k at 90 m from RANS runs instead of the analytic log law")
    ap.add_argument("--out", default=rp("data/dataset.csv"))
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    rans = pd.read_csv(a.rans_csv) if a.rans_csv else None
    rows, t0 = [], time.time()
    for i in range(a.n):
        if rans is not None:
            r = rans.iloc[rng.integers(len(rans))]
            ustar, z0, U = r.ustar, r.z0, r.U90
            sigma_ratio_base = np.sqrt(r.k90 / 0.945) / ustar   # sigma_u implied by RANS k (anisotropy 1:0.8:0.5)
            sigma_ratio = sigma_ratio_base * rng.uniform(1.0, 1.5)   # RANS under-predicts sigma_u -> randomised correction
        else:
            ustar = rng.uniform(0.25, 0.9); z0 = 10 ** rng.uniform(-2, 0); U = log_wind(ustar, z0, a.z)
            sigma_ratio = rng.uniform(1.9, 2.9)
        sigma_u = sigma_ratio * ustar
        L_u = iec_length_scale(a.z) * rng.lognormal(0.0, 0.3)
        L_eff = rng.uniform(*a.leff_range)
        noise = rng.uniform(*a.noise_range)
        u, v = synth_series(U, sigma_u, L_u, rng=rng)
        s = sonic_stats(U, u, v)
        l = lidar_stats(U, u, v, L_eff, Ts=a.Ts, noise_std=noise, rng=rng)
        rows.append(dict(U_lidar=l["mean"], TI_lidar=l["ti"], GF_lidar=l["gf"], U_sonic=s["mean"], TI_sonic=s["ti"],
                         GF_sonic=s["gf"], ustar=ustar, z0=z0, sigma_ratio=sigma_ratio, L_u=L_u, L_eff=L_eff, noise=noise))
        if (i + 1) % 500 == 0:
            print("%d/%d  (%.1f s)" % (i + 1, a.n, time.time() - t0), flush=True)
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    df.to_csv(a.out, index=False)
    per_case = (time.time() - t0) / a.n
    json.dump(dict(n=a.n, z=a.z, Ts=a.Ts, gate=a.gate, seconds_per_case=per_case, seed=a.seed, rans_csv=a.rans_csv),
              open(os.path.splitext(a.out)[0] + "_meta.json", "w"), indent=1)
    print("wrote %s (%d rows, %.3f s per case)" % (a.out, len(df), per_case))
    print(df[["U_lidar", "TI_lidar", "GF_lidar", "TI_sonic", "GF_sonic"]].describe().loc[["mean", "std", "min", "max"]].round(3))

if __name__ == "__main__":
    main()
