"""Simple emulators of a sonic anemometer and a scanning Doppler lidar, based on Taylor's frozen turbulence.

Why a transfer-function model: it is cheap enough to generate thousands of labelled records for training a network, and it needs
only one assumed number, the effective averaging length L_eff. Sonic: 20 Hz speed, gust = maximum 3 s moving mean in 10 min (WMO).
Lidar: spatial average over L_eff, one value per scan cycle, additive noise. This is NOT a beam-level simulation; that is
what lidar/virtual.py is for. Because L_eff is unknown I randomise it in training (domain randomisation) instead of fixing it.
"""
import numpy as np

def _hspeed(U, u, v):
    return np.sqrt((U + u) ** 2 + v ** 2)

def sonic_stats(U, u, v, fs=20.0, gust_window_s=3.0):
    s = _hspeed(U, u, v)
    w = int(round(gust_window_s * fs))
    mov = np.convolve(s, np.ones(w) / w, mode="valid")
    mean = s.mean()
    return dict(mean=mean, ti=s.std() / mean, gust=mov.max(), gf=mov.max() / mean)

def lidar_stats(U, u, v, L_eff, Ts=3.4, fs=20.0, noise_std=0.0, rng=None):
    n = len(u); f = np.fft.rfftfreq(n, 1.0 / fs)
    H = np.sinc(f * L_eff / U) * np.sinc(f * Ts)          # boxcar in space (advected by U) x boxcar over one scan cycle
    uf = np.fft.irfft(np.fft.rfft(u) * H, n); vf = np.fft.irfft(np.fft.rfft(v) * H, n)
    s = _hspeed(U, uf, vf)
    step = max(1, int(round(Ts * fs)))
    s = s[::step]
    if noise_std > 0:
        rng = np.random.default_rng() if rng is None else rng
        s = s + rng.normal(0.0, noise_std, size=s.shape)
    mean = s.mean()
    return dict(mean=mean, ti=s.std() / mean, gust=s.max(), gf=s.max() / mean)
