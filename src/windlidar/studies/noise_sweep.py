"""How much radial-velocity noise does the virtual lidar need so that its gust factor matches the real lidar-minus-sonic bias (+0.04)?

I ran this because the real lidar gust factor is HIGHER than the sonic one, while my first emulator made it lower. A noisy series has a biased-high
maximum, so noise is the obvious candidate. The match is a consistency check, not proof: other error sources look the same.
"""
import sys, os, json
from windlidar.paths import rp
import numpy as np
from windlidar.lidar import virtual as V
from multiprocessing import Pool
NV, PTS, NS = V.geometry()

def run(args):
    sr, seed = args; rng = np.random.default_rng(seed)
    U = rng.uniform(4, 10); ti = rng.uniform(0.08, 0.2)
    F = V.field(U, ti * U, 340 * rng.lognormal(0, .3), PTS, rng); vr = V.scan(F, U, NV, NS, sigma_r=sr, rng=rng); u, v, w = V.retrieve(vr, NV)
    s = np.hypot(u, v); ss = V.sonic_gust(F, U)
    return s.max() / s.mean() - ss["gf"], s.std() / s.mean() - ss["ti"]

if __name__ == "__main__":
    k = np.sqrt(np.linalg.inv(NV.T @ NV)[0, 0]); res = {}
    with Pool(16) as p:
        for sr in (0.1, 0.3, 0.5, 0.8, 1.2, 1.6):
            r = np.array(p.map(run, [(sr, i) for i in range(160)]))
            res[sr] = dict(speed_noise=float(sr * k), gf_bias=float(r[:, 0].mean()), ti_bias=float(r[:, 1].mean()))
            print("sigma_r %.1f (speed noise %.2f m/s): GF bias lidar-sonic %+.3f   TI bias %+.3f" % (sr, sr * k, res[sr]["gf_bias"], res[sr]["ti_bias"]), flush=True)
    json.dump(res, open(rp("results/analysis/noise_sweep.json"), "w"), indent=1)
