"""Ablation: does a stability-aware synthetic training set close the sim-to-real gap?

Why I ran this: after calibrating the lidar noise (see gustnet/dataset.py) the network still explained only part of the
real gust-factor variance. My hypothesis was that the neutral-only emulator was the cause, because the real gust factor is
about 1.44 by day and 1.24 at night. So I trained the same network on emulators with and without each ingredient
(Monin-Obukhov stability, sonic/lidar decoherence, stability-dependent noise) and evaluated every one on the same
chronological real-data hold-out. Outcome: stability did not help (see README), so this script documents a negative result.

Variants: full | no_decoherence | neutral_only | neutral_no_decoh | const_noise.   Output: results/stability_ablation.json
Usage: python -m windlidar.studies.stability_ablation
"""
import json, os, subprocess, sys
import pandas as pd
from windlidar.paths import rp
from windlidar.gustnet.dataset_stability import generate
from windlidar.fesstval.analysis import run

SPLIT = "2021-07-15"          # same chronological split as every other real-data evaluation


def main():
    P = json.load(open(rp("results/stability_emulator_params.json")))["params"]
    R = pd.read_csv(rp("results/fesstval/matched_dwd.csv"), parse_dates=["t"])
    hours = R[R.t < SPLIT].hour.values          # only training-period hours are used to draw emulator conditions (no test leakage)
    variants = {"full": P,
                "no_decoherence": {**P, "d": 0.0},
                "neutral_only": {**P, "H_day": 0.0, "H_night": 0.0},
                "neutral_no_decoh": {**P, "H_day": 0.0, "H_night": 0.0, "d": 0.0},
                "const_noise": {**P, "noise": P["noise"] + P.get("noise_rel", 0) * 1.0, "noise_rel": 0.0}}
    out = {}
    for name, Pv in variants.items():
        mdir, rdir = rp(f"outputs/stability_ablation/models/{name}"), rp(f"outputs/stability_ablation/results/{name}")
        os.makedirs(mdir, exist_ok=True); os.makedirs(rdir, exist_ok=True)
        data = rp(f"outputs/stability_ablation/dataset_{name}.csv")
        generate(8000, Pv, hours, seed=3).to_csv(data, index=False)
        for feats in ("level2_hour", "level2"):
            subprocess.run([sys.executable, "-m", "windlidar.gustnet.train", "--features", feats, "--data", data, "--out", mdir, "--results", rdir],
                           capture_output=True, check=True)
            for prod in ("dwd", "uzk"):
                M = pd.read_csv(rp(f"results/fesstval/matched_{prod}.csv"), parse_dates=["t"])
                r, _, _ = run(M, pd.Timestamp(SPLIT), os.path.join(mdir, f"best_{feats}.pt"))
                nn = "synthetic_nn"
                out[f"{name}|{feats}|{prod}"] = dict(holdout=r["holdout"][nn], day=r["holdout_day"][nn], night=r["holdout_night"][nn], cv=r["blocked_cv"][nn],
                    raw=r["holdout"]["raw"], lin=r["holdout"]["lin_gf"], gbr=r["holdout"]["gbr_diurnal"],
                    raw_cv=r["blocked_cv"]["raw"], lin_cv=r["blocked_cv"]["lin_gf"], gbr_cv=r["blocked_cv"]["gbr_diurnal"])
        print(name, "done", flush=True)
    json.dump(out, open(rp("results/stability_ablation.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
