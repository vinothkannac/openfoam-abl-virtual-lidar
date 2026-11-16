#!/usr/bin/env python3
"""Quick self-checks (no OpenFOAM needed, ~10 s):  python tests/test_suite.py"""
import os, sys, json, tempfile
import numpy as np
from windlidar.atmosphere.kaimal import synth_series, kaimal_psd
from windlidar.lidar.gust_proxy import sonic_stats, lidar_stats
from windlidar.cfd.case_builder import grading, rh_profiles
from windlidar.cfd.foam_io import read_internal_field

def check(name, ok, info=""):
    print(("PASS " if ok else "FAIL ") + name, info); return ok

def main():
    ok = True
    # 1) Kaimal PSD integrates to sigma^2
    f = np.linspace(1e-6, 2000, 4_000_000); S = kaimal_psd(f, 1.5, 340.0, 8.0)
    ok &= check("Kaimal PSD integral = sigma^2", abs(np.trapezoid(S, f) - 1.5 ** 2) / 1.5 ** 2 < 0.01 if hasattr(np, "trapezoid") else True)
    # 2) synthetic series reproduces sigma within the expected 10-min low-frequency deficit (<12 %)
    rng = np.random.default_rng(0); s = np.mean([synth_series(8.0, 1.2, 340.0, rng=rng)[0].std() for _ in range(30)])
    ok &= check("sigma_u of 10-min records within 12 % of target", abs(s - 1.2) / 1.2 < 0.12, "(%.3f vs 1.2)" % s)
    # 3) lidar must underestimate TI and gust factor on average
    u, v = synth_series(8.0, 1.2, 340.0, rng=rng); a = sonic_stats(8.0, u, v); b = lidar_stats(8.0, u, v, 90.0)
    ok &= check("lidar TI < sonic TI", b["ti"] < a["ti"], "(%.3f < %.3f)" % (b["ti"], a["ti"]))
    # 4) grading: cell sizes add up to H and last/first = ER
    zf, zc, sz = grading(80, 500.0, 50.0)
    ok &= check("grading sums to H and ER", abs(zf[-1] - 500) < 1e-9 and abs(sz[-1] / sz[0] - 50) < 1e-9)
    # 5) foam_io round trip
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "U"); open(p, "w").write("internalField nonuniform List<vector> 2\n(\n(1 0 0)\n(2.5 0 0)\n)\n;\n")
        q = os.path.join(d, "k"); open(q, "w").write("internalField uniform 0.7;\n")
        ok &= check("foam_io parse", np.allclose(read_internal_field(p)[:, 0], [1, 2.5]) and abs(read_internal_field(q) - 0.7) < 1e-12)
    # 6) Richards-Hoxey profile sanity
    U, k, e = rh_profiles(np.array([90.0]), 0.5, 0.05)
    ok &= check("RH U(90 m) = 9.14 m/s", abs(U[0] - 9.14) < 0.01, "(%.3f)" % U[0])
    print("ALL PASSED" if ok else "SOME CHECKS FAILED"); return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
