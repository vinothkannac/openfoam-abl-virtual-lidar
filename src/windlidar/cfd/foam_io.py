"""Minimal reader for ASCII OpenFOAM field files (internalField only).

I wrote this instead of depending on ParaView or fluidfoam so that post-processing needs nothing but numpy.
"""
import os, re
import numpy as np

def read_internal_field(path):
    txt = open(path).read()
    m = re.search(r"internalField\s+uniform\s+([^;]+);", txt)
    if m:
        vals = re.findall(r"[-+0-9.eE]+", m.group(1))
        return float(vals[0]) if len(vals) == 1 else np.array([float(v) for v in vals])
    m = re.search(r"internalField\s+nonuniform\s+List<(\w+)>\s*(\d+)\s*\(", txt)
    if not m:
        raise ValueError("cannot parse internalField in %s" % path)
    kind, n = m.group(1), int(m.group(2))
    i, depth, start = m.end(), 1, m.end()
    while depth > 0:
        c = txt[i]
        depth += (c == "(") - (c == ")")
        i += 1
    body = txt[start:i - 1]
    nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", body)
    arr = np.array([float(x) for x in nums])
    arr = arr.reshape(n, 3) if kind == "vector" else arr
    if len(arr) != n:
        raise ValueError("expected %d entries, got %d" % (n, len(arr)))
    return arr

def latest_time_dir(case):
    best, bv = None, -1.0
    for d in os.listdir(case):
        try:
            v = float(d)
        except ValueError:
            continue
        if v > bv and os.path.isdir(os.path.join(case, d)):
            best, bv = d, v
    if best is None or bv <= 0:
        raise SystemExit("no result time directory found in %s (did simpleFoam run?)" % case)
    return best
