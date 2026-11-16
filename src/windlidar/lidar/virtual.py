"""Virtual Doppler lidar in a spatially coherent turbulent field.

Why I built it: the point-wise emulator (gust_proxy.py) cannot say what the scanning geometry itself does to the signal, and that is the
actual signal-processing question. So I synthesise 3-D turbulence at the points the lidar samples and then scan it.
Method: Veers spectral synthesis with the IEC 61400-1 exponential coherence, Taylor frozen turbulence for the downwind phase shift.
Geometry follows the FESSTVaL CSM mode: 62 deg elevation, 10 beams per 3.4 s circulation, 30 m range gate (boxcar over n_sub points).
  cross-spectrum  S_ij(f) = S(f) Coh(f, d_perp) exp(i 2 pi f dx / U),   Coh = exp(-a sqrt((f d/U)^2 + (0.12 d/Lc)^2)),  a = 12
Retrieval: least-squares fit of V = (u, v, w) to the 10 radial velocities of one circulation (VAD); horizontal speed = sqrt(u^2 + v^2).
Simplifications: neutral stationary Gaussian turbulence, no beam-pointing error, no pulse-weighting function (boxcar gate instead), no cloud/CNR
effects, noise is white and Gaussian. The field is generated at 10 Hz.
"""
import os, sys
import numpy as np
from windlidar.atmosphere.kaimal import kaimal_psd

FS = 10.0
LC = 8.1 * 42.0
CP = dict(a=12.0, lam=42.0)          # IEC: Lambda_1 = 42 m above 60 m


def geometry(z0=90.0, elev_deg=62.0, n_beams=10, gate=30.0, n_sub=3):
    e = np.deg2rad(elev_deg); r0 = z0 / np.sin(e)
    az = 2 * np.pi * np.arange(n_beams) / n_beams
    nvec = np.stack([np.cos(e) * np.cos(az), np.cos(e) * np.sin(az), np.full(n_beams, np.sin(e))], 1)   # beam unit vectors (x downwind)
    offs = (np.arange(n_sub) + 0.5) / n_sub * gate - gate / 2
    pts = [(r0 + o) * nvec[k] for k in range(n_beams) for o in offs]
    pts = np.vstack([np.array([[0.0, 0.0, z0]])] + [np.array(p)[None] for p in pts])                  # row 0 = sonic point
    return nvec, pts, n_sub


def _chol(U, pts, n, fs, Lcoh=LC, a=12.0):
    f = np.fft.rfftfreq(n, 1.0 / fs)[1:]
    dx = pts[:, None, 0] - pts[None, :, 0]
    dperp = np.hypot(pts[:, None, 1] - pts[None, :, 1], pts[:, None, 2] - pts[None, :, 2])
    coh = np.exp(-a * np.sqrt((f[:, None, None] * dperp / U) ** 2 + (0.12 * dperp / Lcoh) ** 2))
    G = coh * np.exp(1j * 2 * np.pi * f[:, None, None] * dx / U)
    G = G + 1e-6 * np.eye(len(pts))
    return f, np.linalg.cholesky(G)


def field(U, sigma_u, L_u, pts, rng, T=600.0, fs=FS, ratios=(1.0, 0.8, 0.5), lens=(1.0, 1 / 3, 1 / 12)):
    """u,v,w series, shape (3, N, n) at all points (mean wind U along x)."""
    n = int(round(T * fs)); df = 1.0 / T
    f, H = _chol(U, pts, n, fs)
    out = []
    for rs, ls in zip(ratios, lens):
        S = kaimal_psd(f, rs * sigma_u, ls * L_u, U)
        xi = (rng.standard_normal((len(f), len(pts))) + 1j * rng.standard_normal((len(f), len(pts)))) / np.sqrt(2)
        X = np.einsum("fij,fj->fi", H, xi) * np.sqrt(S * df)[:, None] * n / np.sqrt(2)
        Xf = np.zeros((n // 2 + 1, len(pts)), complex); Xf[1:] = X
        out.append(np.fft.irfft(Xf, n, axis=0).T)
    return np.stack(out)


def scan(F, U, nvec, n_sub, Ts=3.4, fs=FS, sigma_r=0.2, rng=None):
    """Return radial velocities (n_cycles, n_beams) from field F (3,N,n)."""
    nb = len(nvec); n = F.shape[2]; nc = int((n / fs - Ts) // Ts) + 1
    vr = np.zeros((nc, nb))
    for m in range(nc):
        for k in range(nb):
            idx = min(n - 1, int(round((m * Ts + (k + 0.5) * Ts / nb) * fs)))
            pk = 1 + k * n_sub + np.arange(n_sub)
            V = np.stack([U + F[0, pk, idx], F[1, pk, idx], F[2, pk, idx]], 1).mean(0)
            vr[m, k] = nvec[k] @ V
    if sigma_r > 0:
        vr += rng.normal(0, sigma_r, vr.shape)
    return vr


def retrieve(vr, nvec):
    Vh = np.linalg.pinv(nvec) @ vr.T          # (3, n_cycles)
    return Vh[0], Vh[1], Vh[2]


def sonic_gust(F, U, fs=FS, win_s=3.0):
    s = np.hypot(U + F[0, 0], F[1, 0]); w = int(round(win_s * fs))
    mov = np.convolve(s, np.ones(w) / w, "valid")
    return dict(mean=s.mean(), sigma=s.std(), ti=s.std() / s.mean(), gf=mov.max() / s.mean())


def lidar_stats(U_l, V_l):
    s = np.hypot(U_l, V_l)
    return dict(mean=s.mean(), sigma=s.std(), ti=s.std() / s.mean(), gf=s.max() / s.mean(), series=s)
