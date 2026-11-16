"""Stability-aware emulator: Monin-Obukhov similarity plus decorrelation between the sonic and the lidar volume.

Why I wrote it: the real gust factor at Falkenberg is about 1.44 by day and 1.24 at night, which a neutral emulator cannot produce.
I used only textbook relations with few free parameters so that they can be calibrated on the training period alone:
  * surface heat flux from hour of day (sun elevation proxy and a cloud factor), then zeta = z/L coupled to u*
  * Businger-Dyer wind profile, Panofsky-Dutton sigma_u/u*, stability-dependent length scale
  * the lidar does not sample the sonic's air: lidar = c(f) X_sonic + sqrt(1-c(f)^2) X_indep with Davenport coherence c(f) = exp(-a d f / U)
Simplifications: no sensible-heat-flux measurement is used (hour of day is a proxy), the boundary-layer height is fixed at 1200 m, and the
coherence model is a one-parameter fit. The ablation in studies/stability_ablation.py shows this emulator did NOT improve the real-data result.
"""
import numpy as np
from windlidar.atmosphere.kaimal import kaimal_psd, KAPPA, iec_length_scale

G, THETA0, ZI = 9.81, 290.0, 1200.0
SUNRISE, SUNSET = 4.0, 19.0          # UTC hours, May-Aug average at Lindenberg (approximate)


def sun(hour):
    h = hour + 0.5
    return float(np.sin(np.pi * (h - SUNRISE) / (SUNSET - SUNRISE))) if SUNRISE < h < SUNSET else 0.0


def psi_m(zeta):
    if zeta < 0:
        x = (1 - 16 * zeta) ** 0.25
        return 2 * np.log((1 + x) / 2) + np.log((1 + x * x) / 2) - 2 * np.arctan(x) + np.pi / 2
    return -5.0 * zeta


def sample_state(rng, hour, z, P):
    """Draw one physical state for a given hour. P: dict with H_day, H_night (K m/s)."""
    ustar = rng.uniform(0.15, 0.85); z0 = 10 ** rng.uniform(-2.0, -0.5)
    cloud = rng.uniform(0.15, 1.0)
    s = sun(hour)
    Hs = P["H_day"] * s * cloud if s > 0.05 else -P["H_night"] * rng.uniform(0.2, 1.0)
    zeta = -KAPPA * G * z * Hs / (THETA0 * ustar ** 3)
    zeta = float(np.clip(zeta, -5.0, 2.0))
    U = ustar / KAPPA * (np.log(z / z0) - psi_m(zeta))
    if zeta < 0:
        sr = (12 + 0.5 * ZI / (z / abs(zeta) + 1e-9)) ** (1 / 3)
        Lf = 1 + 0.5 * min(-zeta, 4.0)
    else:
        sr = 2.29 * np.exp(-0.15 * zeta); Lf = 1 / (1 + 0.5 * zeta)
    sr *= rng.lognormal(0, P.get("sig_spread", 0.10))
    L_u = iec_length_scale(z) * Lf * rng.lognormal(0, 0.3)
    return dict(ustar=ustar, z0=z0, Hs=Hs, zeta=zeta, U=float(U), sigma_u=sr * ustar, sigma_ratio=sr, L_u=L_u)


def _coef(n, fs, T, S, rng):
    f_df = 1.0 / T
    amp = np.sqrt(2.0 * S * f_df)
    return (n / 2.0) * amp * np.exp(1j * rng.uniform(0, 2 * np.pi, len(S)))


def synth_pair(U, sigma_u, L_u, d, fs=20.0, T=600.0, rng=None, v_ratio=0.8, v_len_ratio=1.0 / 3.0, a=7.0):
    """Return (u_s, v_s) seen by the sonic and (u_l, v_l) the turbulence the lidar volume sees (coherent with the sonic only at low f)."""
    n = int(round(fs * T)); f = np.fft.rfftfreq(n, 1.0 / fs)
    c = np.exp(-a * d * f / max(U, 0.5)); s2 = np.sqrt(np.clip(1 - c ** 2, 0, 1))
    out = []
    for sig, L in ((sigma_u, L_u), (v_ratio * sigma_u, v_len_ratio * L_u)):
        S = np.zeros_like(f); S[1:] = kaimal_psd(f[1:], sig, L, U)
        X1 = _coef(n, fs, T, S, rng); X2 = _coef(n, fs, T, S, rng)
        X1[0] = X2[0] = 0.0
        out.append((np.fft.irfft(X1, n), np.fft.irfft(c * X1 + s2 * X2, n)))
    return out[0][0], out[1][0], out[0][1], out[1][1]
