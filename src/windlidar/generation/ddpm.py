"""Conditional diffusion model (DDPM, 1-D U-Net) for 64 s turbulent wind series at 10 Hz, normalised by sigma_u.

Why a diffusion model: it learns a distribution of time series directly and can be conditioned on a gust, which fits the "AI for wind-field
generation" question. Conditions: L_u/U (spectral shape), a gust flag, and the gust amplitude A = Vgust/sigma (IEC EOG shape at a random time).
Honest limitation: I trained it on Kaimal-plus-EOG data, for which an exact analytic generator already exists, so it can at best match that
baseline (it is slightly worse, see README). Its real value would be training on measured, non-Gaussian data, which I have not done.
Usage: python -m windlidar.generation.ddpm train --steps 6000   |   python -m windlidar.generation.ddpm eval
"""
from windlidar.paths import rp
import argparse, json, math, os, sys, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from windlidar.atmosphere.kaimal import synth_series

FS, N, T_EOG = 10.0, 640, 10.5
OUT = rp("results/analysis"); os.makedirs(OUT, exist_ok=True)
CK = os.path.join(OUT, "diffusion_wind.pt")


def eog_template():
    t = np.arange(0, T_EOG, 1 / FS); tau = t / T_EOG
    return -0.37 * np.sin(3 * np.pi * tau) * (1 - np.cos(2 * np.pi * tau))   # unit Vgust


def make_windows(n_rec, seed, p_gust=0.4):
    rng = np.random.default_rng(seed); tpl = eog_template(); X, C = [], []
    for _ in range(n_rec):
        U = rng.uniform(4, 14); L = 340.0 * rng.lognormal(0, 0.3)
        u, _ = synth_series(U, 1.0, L, fs=FS, T=600.0, rng=rng)
        for k in range(9):
            w = u[k * N:(k + 1) * N].copy(); flag = float(rng.random() < p_gust); A = 0.0
            if flag:
                A = rng.uniform(2.0, 6.0); on = rng.integers(20, N - len(tpl) - 20); w[on:on + len(tpl)] += A * tpl
            X.append(w); C.append([np.log(L / U) - 3.9, flag, A / 6.0])
    return np.array(X, np.float32), np.array(C, np.float32)


class Block(nn.Module):
    def __init__(s, ci, co, ed):
        super().__init__(); s.n1 = nn.GroupNorm(8, ci); s.c1 = nn.Conv1d(ci, co, 5, padding=2); s.e = nn.Linear(ed, co)
        s.n2 = nn.GroupNorm(8, co); s.c2 = nn.Conv1d(co, co, 5, padding=2); s.sk = nn.Conv1d(ci, co, 1) if ci != co else nn.Identity()
    def forward(s, x, e):
        h = s.c1(F.silu(s.n1(x))) + s.e(e)[:, :, None]; h = s.c2(F.silu(s.n2(h))); return h + s.sk(x)


class UNet(nn.Module):
    def __init__(s, ed=128, ch=(24, 48, 64, 96)):
        super().__init__(); s.ed = ed
        s.emb = nn.Sequential(nn.Linear(ed, ed), nn.SiLU(), nn.Linear(ed, ed)); s.cemb = nn.Sequential(nn.Linear(3, ed), nn.SiLU(), nn.Linear(ed, ed))
        s.inp = nn.Conv1d(1, ch[0], 5, padding=2)
        s.down = nn.ModuleList([Block(ch[i], ch[i + 1], ed) for i in range(3)]); s.mid = Block(ch[3], ch[3], ed)
        s.up = nn.ModuleList([Block(ch[i + 1] + ch[i + 1], ch[i], ed) for i in (2, 1, 0)]); s.out = nn.Conv1d(ch[0], 1, 5, padding=2)
    def temb(s, t):
        h = s.ed // 2; f = torch.exp(-math.log(10000) * torch.arange(h, device=t.device) / h); a = t[:, None].float() * f[None]
        return torch.cat([a.sin(), a.cos()], 1)
    def forward(s, x, t, c):
        e = s.emb(s.temb(t)) + s.cemb(c); h = s.inp(x[:, None]); hs = []
        for b in s.down: h = b(h, e); hs.append(h); h = F.avg_pool1d(h, 2)
        h = s.mid(h, e)
        for b in s.up: h = F.interpolate(h, scale_factor=2); h = b(torch.cat([h, hs.pop()], 1), e)
        return s.out(F.silu(h))[:, 0]


TS = 200
def schedule():
    b = torch.linspace(1e-4, 0.03, TS); a = 1 - b; ab = torch.cumprod(a, 0); return b, a, ab


def train(steps, bs=64):
    torch.set_num_threads(16); torch.manual_seed(0)
    X, C = make_windows(1500, 1); print("train windows", X.shape, flush=True)
    X, C = torch.tensor(X), torch.tensor(C); net = UNet(); opt = torch.optim.AdamW(net.parameters(), 2e-4, weight_decay=1e-5)
    b, a, ab = schedule(); t0 = time.time(); sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    for s in range(steps):
        i = torch.randint(0, len(X), (bs,)); x0, c = X[i], C[i]; t = torch.randint(0, TS, (bs,)); eps = torch.randn_like(x0)
        xt = ab[t].sqrt()[:, None] * x0 + (1 - ab[t]).sqrt()[:, None] * eps
        loss = F.mse_loss(net(xt, t, c), eps); opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); sched.step()
        if s % 250 == 0: print(f"step {s}  loss {loss.item():.4f}  {time.time() - t0:.0f}s", flush=True)
        if s % 1000 == 999: torch.save(net.state_dict(), CK)
    torch.save(net.state_dict(), CK)


@torch.no_grad()
def sample(net, c, seed=0):
    g = torch.Generator().manual_seed(seed); b, a, ab = schedule(); x = torch.randn(len(c), N, generator=g)
    for t in reversed(range(TS)):
        tt = torch.full((len(c),), t); eps = net(x, tt, c)
        x = (x - b[t] / (1 - ab[t]).sqrt() * eps) / a[t].sqrt()
        if t > 0: x = x + ((1 - ab[t - 1]) / (1 - ab[t]) * b[t]).sqrt() * torch.randn(x.shape, generator=g)
    return x.numpy()


def psd(X):
    w = np.hanning(N); P = np.abs(np.fft.rfft((X - X.mean(1, keepdims=True)) * w, axis=1)) ** 2 / (np.sum(w ** 2) * FS); return np.fft.rfftfreq(N, 1 / FS), P.mean(0)


def kurt(x): x = x - x.mean(); return float((x ** 4).mean() / (x ** 2).mean() ** 2)


def gust_amp(X):
    tpl = eog_template(); m = len(tpl); sm = np.stack([np.convolve(x, np.ones(5) / 5, "same") for x in X])
    est = []
    for x in sm:
        best = 0.0
        for on in range(10, N - m - 10, 2):
            seg = x[on:on + m]; seg = seg - seg.mean(); tt = tpl - tpl.mean(); best = max(best, float(seg @ tt / (tt @ tt)))
        est.append(best)
    return np.array(est)


def evaluate():
    torch.set_num_threads(16); net = UNet(); net.load_state_dict(torch.load(CK)); net.eval()
    res = {}; rng = np.random.default_rng(5)
    for name, flag, A in [("normal", 0, 0.0), ("EOG_A=3", 1, 3.0), ("EOG_A=5", 1, 5.0)]:
        # held-out conditions drawn from the training range of L/U
        LU = rng.uniform(25, 90, 256); c = np.stack([np.log(LU) - 3.9, np.full(256, flag), np.full(256, A / 6.0)], 1).astype(np.float32)
        gen = sample(net, torch.tensor(c), seed=1)
        # reference: Kaimal + analytic EOG at the same conditions
        ref = []; tpl = eog_template()
        for lu in LU:
            u, _ = synth_series(1.0, 1.0, lu * 1.0, fs=FS, T=64.0, rng=rng); w = u[:N]
            if flag: on = rng.integers(20, N - len(tpl) - 20); w[on:on + len(tpl)] += A * tpl
            ref.append(w)
        ref = np.array(ref); f, Pg = psd(gen); _, Pr = psd(ref)
        bands = {"0.05-0.2Hz": (0.05, 0.2), "0.2-1Hz": (0.2, 1.0), "1-5Hz": (1.0, 5.0)}
        r = {"std_gen": float(gen.std()), "std_ref": float(ref.std()),
             "psd_ratio_gen_over_ref": {k: float(np.exp(np.mean(np.log(Pg[(f >= lo) & (f < hi)] / Pr[(f >= lo) & (f < hi)])))) for k, (lo, hi) in bands.items()},
             "increment_kurtosis_1s": dict(gen=kurt((gen[:, 10:] - gen[:, :-10]).ravel()), ref=kurt((ref[:, 10:] - ref[:, :-10]).ravel())),
             "increment_kurtosis_0.1s": dict(gen=kurt(np.diff(gen, axis=1).ravel()), ref=kurt(np.diff(ref, axis=1).ravel()))}
        if flag:
            ag, ar = gust_amp(gen), gust_amp(ref); r["gust_amplitude_estimate"] = dict(requested=A, gen_mean=float(ag.mean()), gen_sd=float(ag.std()), ref_mean=float(ar.mean()), ref_sd=float(ar.std()))
        res[name] = r; print(name, json.dumps(r), flush=True)
        if name == "EOG_A=5":
            import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
            fig, ax = plt.subplots(1, 3, figsize=(15, 4)); tt = np.arange(N) / FS
            ax[0].plot(tt, gen[0], lw=.8, label="diffusion model"); ax[0].plot(tt, ref[0] - 4, lw=.8, label="Kaimal + EOG (baseline)"); ax[0].legend(fontsize=7); ax[0].set_xlabel("t [s]"); ax[0].set_title("example, A = 5 sigma EOG")
            ax[1].loglog(f[1:], Pg[1:], label="diffusion"); ax[1].loglog(f[1:], Pr[1:], label="Kaimal baseline"); ax[1].set_xlabel("f [Hz]"); ax[1].set_ylabel("PSD"); ax[1].legend(); ax[1].set_title("spectra")
            ax[2].hist([ag, ar], bins=20, label=["diffusion", "Kaimal+EOG"]); ax[2].axvline(A, c="k", ls="--"); ax[2].set_title("recovered gust amplitude (A/sigma)"); ax[2].legend()
            fig.tight_layout(); fig.savefig(os.path.join(OUT, "diffusion_samples.png"), dpi=130)
    json.dump(res, open(os.path.join(OUT, "ddpm_eval.json"), "w"), indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("mode"); ap.add_argument("--steps", type=int, default=6000); a = ap.parse_args()
    train(a.steps) if a.mode == "train" else evaluate()
