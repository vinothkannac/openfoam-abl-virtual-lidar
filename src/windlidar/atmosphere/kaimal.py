"""Kaimal turbulence spectra and random-phase synthesis of 10-minute wind records.

Why Kaimal: it is the spectrum used by IEC 61400-1 for the normal turbulence model, it has a single length scale and a single
variance, and it is what a wind-energy reviewer expects as the "neutral reference". Simplification: I treat the turbulence as
stationary, Gaussian and neutral, and I use Taylor's frozen-turbulence hypothesis to convert time series to space. Real boundary-layer
turbulence is neither Gaussian nor stationary, which is one reason the simulated and measured gust factors differ.
"""
import numpy as np

KAPPA = 0.41

def iec_length_scale(z):
    """Longitudinal integral length scale L_u = 8.1 * Lambda_u (IEC 61400-1): Lambda_u = 0.7 z (z<60 m) else 42 m."""
    return 8.1 * (0.7 * z if z < 60 else 42.0)

def kaimal_psd(f, sigma, L, U):
    """One-sided PSD with  integral_0^inf S df = sigma^2."""
    return 4 * sigma ** 2 * (L / U) / (1 + 6 * f * L / U) ** (5.0 / 3.0)

def synth_series(U, sigma_u, L_u, fs=20.0, T=600.0, rng=None, v_ratio=0.8, v_len_ratio=1.0 / 3.0):
    """Return fluctuations u (along mean wind), v (cross wind), each of length fs*T, zero mean over the record."""
    rng = np.random.default_rng() if rng is None else rng
    n = int(round(fs * T)); f = np.fft.rfftfreq(n, 1.0 / fs); df = 1.0 / T
    def comp(sig, L):
        S = np.zeros_like(f); S[1:] = kaimal_psd(f[1:], sig, L, U)
        amp = np.sqrt(2.0 * S * df)
        X = (n / 2.0) * amp * np.exp(1j * rng.uniform(0, 2 * np.pi, len(f)))
        X[0] = 0.0
        return np.fft.irfft(X, n)
    return comp(sigma_u, L_u), comp(v_ratio * sigma_u, v_len_ratio * L_u)

def log_wind(ustar, z0, z):
    return ustar / KAPPA * np.log(z / z0)
