"""Collect the metrics.json of every run in a sweep directory into one CSV (u*, z0, U90, k90, eps90).

I keep this separate from the post-processing so that the 16-run sweep can be re-collected without re-running OpenFOAM.
Usage: python -m windlidar.cfd.collect_sweep <sweep_dir> <out.csv>
"""
import glob, json, os, sys
import numpy as np
from windlidar.paths import rp


def main():
    sweep = sys.argv[1] if len(sys.argv) > 1 else rp("outputs/sweep")
    out = sys.argv[2] if len(sys.argv) > 2 else rp("data/rans_sweep.csv")
    rows = []
    for p in sorted(glob.glob(os.path.join(sweep, "*", "metrics.json"))):
        m = json.load(open(p)); rows.append([m["ustar"], m["z0"], m["U90"], m["k90"], m["eps90"]])
    if not rows:                      # never overwrite an existing result with an empty file
        raise SystemExit(f"no metrics.json found in {sweep}")
    np.savetxt(out, np.array(rows), delimiter=",", header="ustar,z0,U90,k90,eps90", comments="")
    print("wrote", out, len(rows), "rows")


if __name__ == "__main__":
    main()
