"""Generate blockMeshDict and the 0/ fields for a horizontally homogeneous, neutral atmospheric boundary layer (RANS k-epsilon).

Richards & Hoxey (1993) profiles:  U = u*/kappa ln((z+z0)/z0),  k = u*^2/sqrt(Cmu),  eps = u*^3/(kappa (z+z0)).
Why: this is the standard verification case for ABL modelling, because the analytic solution is known and any deviation is a numerical or
set-up error. I write the inlet profile as an explicit nonuniform list instead of using atmBoundaryLayer boundary conditions so the
same case runs in OpenFOAM.com and OpenFOAM.org. The wall uses atmNutkWallFunction (it needs libatmosphericModels).
Simplifications: 2-D, flat terrain, no Coriolis force, no stratification, constant k with height (the Richards-Hoxey assumption).
Usage: python -m windlidar.cfd.case_builder --case . --ustar 0.5 --z0 0.05 --nz 80
"""
import argparse, json, os
import numpy as np

KAPPA, CMU = 0.41, 0.09

def grading(nz, H, ER):
    """Geometric grading: ratio last/first cell = ER (same meaning as blockMesh simpleGrading)."""
    q = ER ** (1.0 / (nz - 1)) if nz > 1 else 1.0
    d0 = H * (q - 1) / (q ** nz - 1) if abs(q - 1) > 1e-12 else H / nz
    sizes = d0 * q ** np.arange(nz)
    zf = np.concatenate([[0.0], np.cumsum(sizes)])
    zc = 0.5 * (zf[:-1] + zf[1:])
    return zf, zc, sizes

def rh_profiles(z, ustar, z0):
    U = ustar / KAPPA * np.log((z + z0) / z0)
    k = np.full_like(z, ustar ** 2 / np.sqrt(CMU), dtype=float)
    eps = ustar ** 3 / (KAPPA * (z + z0))
    return U, k, eps

HEADER = """FoamFile
{{
    version     2.0;
    format      ascii;
    class       {cls};
    object      {obj};
}}

"""

def vec_list(values):
    return "nonuniform List<vector> %d\n(\n%s\n)\n;" % (len(values), "\n".join("(%.8g 0 0)" % v for v in values))

def write_field(path, cls, obj, dims, internal, boundary):
    with open(path, "w") as f:
        f.write(HEADER.format(cls=cls, obj=obj))
        f.write("dimensions      %s;\n\ninternalField   %s;\n\nboundaryField\n{\n" % (dims, internal))
        for name, body in boundary.items():
            f.write("    %s\n    {\n" % name)
            for line in body.strip("\n").split("\n"):
                f.write("        %s\n" % line.strip())
            f.write("    }\n")
        f.write("}\n")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default=".")
    ap.add_argument("--ustar", type=float, default=0.5, help="friction velocity [m/s]")
    ap.add_argument("--z0", type=float, default=0.05, help="roughness length [m]")
    ap.add_argument("--nz", type=int, default=80, help="cells in vertical direction")
    ap.add_argument("--nx", type=int, default=20)
    ap.add_argument("--H", type=float, default=500.0, help="domain height [m]")
    ap.add_argument("--L", type=float, default=1000.0, help="domain length [m]")
    ap.add_argument("--ER", type=float, default=50.0, help="total expansion ratio (last/first cell)")
    a = ap.parse_args()

    zf, zc, sizes = grading(a.nz, a.H, a.ER)
    if sizes[0] / 2 <= 1.05 * a.z0:
        raise SystemExit("First cell centre (%.3f m) must be above z0 (%.3f m) for the rough wall function. "
                         "Use fewer cells (--nz), a smaller --ER, or a smaller --z0." % (sizes[0] / 2, a.z0))
    os.makedirs(os.path.join(a.case, "system"), exist_ok=True)
    os.makedirs(os.path.join(a.case, "0"), exist_ok=True)

    # ---- blockMeshDict
    L, H = a.L, a.H
    bmd = HEADER.format(cls="dictionary", obj="blockMeshDict") + f"""convertToMeters 1;

vertices
(
    (0 0 0) ({L} 0 0) ({L} 1 0) (0 1 0)
    (0 0 {H}) ({L} 0 {H}) ({L} 1 {H}) (0 1 {H})
);

blocks
(
    hex (0 1 2 3 4 5 6 7) ({a.nx} 1 {a.nz}) simpleGrading (1 1 {a.ER})
);

edges ();

boundary
(
    inlet  {{ type patch; faces ((0 4 7 3)); }}
    outlet {{ type patch; faces ((1 2 6 5)); }}
    ground {{ type wall;  faces ((0 3 2 1)); }}
    top    {{ type patch; faces ((4 5 6 7)); }}
    frontAndBack {{ type empty; faces ((0 1 5 4) (3 7 6 2)); }}
);

mergePatchPairs ();
"""
    open(os.path.join(a.case, "system", "blockMeshDict"), "w").write(bmd)

    # ---- fields
    U_in, k_in, e_in = rh_profiles(zc, a.ustar, a.z0)
    Utop, ktop, etop = [x[0] for x in rh_profiles(np.array([H]), a.ustar, a.z0)]
    k0 = a.ustar ** 2 / np.sqrt(CMU)
    Uinit = float(rh_profiles(np.array([50.0]), a.ustar, a.z0)[0][0])
    einit = a.ustar ** 3 / (KAPPA * (50.0 + a.z0))
    d = os.path.join(a.case, "0")

    write_field(os.path.join(d, "U"), "volVectorField", "U", "[0 1 -1 0 0 0 0]",
                "uniform (%.6g 0 0)" % Uinit, {
        "inlet": "type fixedValue;\nvalue %s" % vec_list(U_in),
        "outlet": "type zeroGradient;",
        "ground": "type noSlip;",
        "top": "type fixedValue;\nvalue uniform (%.8g 0 0);" % Utop,
        "frontAndBack": "type empty;"})

    write_field(os.path.join(d, "k"), "volScalarField", "k", "[0 2 -2 0 0 0 0]", "uniform %.8g" % k0, {
        "inlet": "type fixedValue;\nvalue uniform %.8g;" % k0,
        "outlet": "type zeroGradient;",
        "ground": "type kqRWallFunction;\nvalue uniform %.8g;" % k0,
        "top": "type fixedValue;\nvalue uniform %.8g;" % ktop,
        "frontAndBack": "type empty;"})
    write_field(os.path.join(d, "epsilon"), "volScalarField", "epsilon", "[0 2 -3 0 0 0 0]", "uniform %.8g" % einit, {
        "inlet": "type fixedValue;\nvalue nonuniform List<scalar> %d\n(\n%s\n)\n;" % (a.nz, "\n".join("%.8g" % e for e in e_in)),
        "outlet": "type zeroGradient;",
        "ground": "type epsilonWallFunction;\nvalue uniform %.8g;" % einit,
        "top": "type fixedValue;\nvalue uniform %.8g;" % etop,
        "frontAndBack": "type empty;"})
    write_field(os.path.join(d, "nut"), "volScalarField", "nut", "[0 2 -1 0 0 0 0]", "uniform 0", {
        "inlet": "type calculated;\nvalue uniform 0;",
        "outlet": "type calculated;\nvalue uniform 0;",
        "ground": "type atmNutkWallFunction;\nz0 uniform %.8g;\nvalue uniform 0;" % a.z0,
        "top": "type calculated;\nvalue uniform 0;",
        "frontAndBack": "type empty;"})
    write_field(os.path.join(d, "p"), "volScalarField", "p", "[0 2 -2 0 0 0 0]", "uniform 0", {
        "inlet": "type zeroGradient;",
        "outlet": "type fixedValue;\nvalue uniform 0;",
        "ground": "type zeroGradient;",
        "top": "type zeroGradient;",
        "frontAndBack": "type empty;"})

    params = dict(ustar=a.ustar, z0=a.z0, nz=a.nz, nx=a.nx, H=a.H, L=a.L, ER=a.ER,
                  dz_first=float(sizes[0]), zc=[float(z) for z in zc])
    json.dump(params, open(os.path.join(a.case, "case_params.json"), "w"), indent=1)
    print("case written to %s: u*=%.3f z0=%.3f nz=%d first cell %.3f m, U(90 m)=%.2f m/s"
          % (a.case, a.ustar, a.z0, a.nz, sizes[0], a.ustar / KAPPA * np.log((90 + a.z0) / a.z0)))

if __name__ == "__main__":
    main()
