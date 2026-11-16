"""Render the project videos (MP4 + GIF) into docs/media.

    python -m windlidar.animations [scan|diffusion|diurnal|all]
"""
import os, subprocess, sys
import numpy as np
import imageio_ffmpeg
from matplotlib import animation
from windlidar import viz
from windlidar.viz import plt, C, INK2, MUTED
from windlidar.paths import rp

FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
plt.rcParams["animation.ffmpeg_path"] = FFMPEG
MEDIA = rp("docs", "media"); os.makedirs(MEDIA, exist_ok=True)


def render(fig, update, frames, name, fps=25, dpi=110):
    anim = animation.FuncAnimation(fig, update, frames=frames, blit=False)
    mp4 = f"{MEDIA}/{name}.mp4"
    anim.save(mp4, writer=animation.FFMpegWriter(fps=fps, codec="libx264", extra_args=["-pix_fmt", "yuv420p", "-crf", "20"]), dpi=dpi)
    gif = f"{MEDIA}/{name}.gif"
    vf = "fps=15,scale=800:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=96[p];[b][p]paletteuse=dither=bayer:bayer_scale=4"
    subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-i", mp4, "-vf", vf, gif], check=True)
    plt.close(fig); print("wrote", mp4, gif)


def scan_video():
    from windlidar.lidar import virtual as V
    rng = np.random.default_rng(11); U, ti, L = 7.0, 0.14, 340.0
    nvec, pts, ns = V.geometry(); F = V.field(U, ti * U, L, pts, rng, T=300.0)
    vr = V.scan(F, U, nvec, ns, sigma_r=0.45, rng=rng); u, v, w = V.retrieve(vr, nvec); s_l = np.hypot(u, v)
    Ts = 3.4; tt = np.arange(F.shape[2]) / V.FS; s_s = np.hypot(U + F[0, 0], F[1, 0])
    nb = len(nvec); az = np.arctan2(nvec[:, 1], nvec[:, 0])
    fig = plt.figure(figsize=(12.8, 5.6)); gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.35], wspace=.12)
    a3 = fig.add_subplot(gs[0], projection="3d"); a2 = fig.add_subplot(gs[1])
    t_end = 120.0; nfr = int(t_end / (Ts / nb)); R = 90 / np.sin(np.radians(62)); rr = np.hypot(*pts[1:4, :2].T)

    def upd(i):
        a3.cla(); a2.cla(); t = i * Ts / nb; k = i % nb; cyc = i // nb
        th = np.linspace(0, 2 * np.pi, 80); rad = 90 / np.tan(np.radians(62))
        a3.plot(rad * np.cos(th), rad * np.sin(th), np.full(80, 90), c=MUTED, lw=.8)
        for kk in range(nb):
            tip = nvec[kk] * R; a3.plot([0, tip[0]], [0, tip[1]], [0, tip[2]], c=viz.GRID, lw=.8)
        tip = nvec[k] * R; a3.plot([0, tip[0]], [0, tip[1]], [0, tip[2]], c=C["orange"], lw=2.2)
        g = pts[1 + k * ns:1 + (k + 1) * ns]; a3.scatter(g[:, 0], g[:, 1], g[:, 2], c=C["orange"], s=28)
        a3.scatter([0], [0], [90], c=C["blue"], s=60, marker="s"); a3.text(0, 0, 100, "sonic", fontsize=8, color=INK2, ha="center")
        a3.scatter([0], [0], [0], c=INK2, marker="^", s=50)
        a3.quiver(-70, -70, 0, 40, 0, 0, color=C["blue"], lw=2, arrow_length_ratio=.25); a3.text(-70, -70, 10, f"U = {U:g} m/s", fontsize=8, color=INK2)
        a3.set(xlim=(-80, 80), ylim=(-80, 80), zlim=(0, 130)); a3.set_box_aspect((1, 1, .8)); a3.view_init(18, 35 + 0.25 * i)
        a3.set_title(f"CSM scan: beam {k + 1}/{nb}, 62° elevation", loc="left")
        a3.xaxis.pane.fill = a3.yaxis.pane.fill = a3.zaxis.pane.fill = False; a3.set_xlabel("x [m]"); a3.set_ylabel("y [m]"); a3.set_zlabel("z [m]")
        m = tt <= t; a2.plot(tt[m], s_s[m], c=C["blue"], lw=.9, label="sonic, 10 Hz (truth at 90 m)")
        nc = cyc; tc = (np.arange(nc) + .5) * Ts
        a2.step(tc, s_l[:nc], where="mid", c=C["orange"], lw=2, label="lidar retrieval (1 per 3.4 s, noisy)")
        a2.axvline(t, c=MUTED, lw=.8); a2.set(xlim=(0, t_end), ylim=(U - 3.2, U + 3.2), xlabel="time  [s]", ylabel="horizontal wind speed  [m/s]", title="Same turbulence, two instruments")
        a2.legend(loc="upper right")
    fig.suptitle("Virtual Doppler lidar in a coherent turbulent field", x=.01, ha="left", fontsize=14, fontweight="bold")
    render(fig, upd, nfr, "virtual_lidar_scan", fps=25, dpi=100)


def diffusion_video():
    import torch
    from windlidar.generation import ddpm as D
    from windlidar.atmosphere.kaimal import synth_series
    net = D.UNet(); net.load_state_dict(torch.load(D.CK)); net.eval()
    b, a, ab = D.schedule(); g = torch.Generator().manual_seed(4)
    c = torch.tensor([[np.log(50.0) - 3.9, 1.0, 5.0 / 6.0]], dtype=torch.float32); x = torch.randn(1, D.N, generator=g); traj = [x[0].numpy().copy()]; ts = [D.TS]
    with torch.no_grad():
        for t in reversed(range(D.TS)):
            eps = net(x, torch.full((1,), t), c); x = (x - b[t] / (1 - ab[t]).sqrt() * eps) / a[t].sqrt()
            if t > 0: x = x + ((1 - ab[t - 1]) / (1 - ab[t]) * b[t]).sqrt() * torch.randn(x.shape, generator=g)
            if t % 4 == 0 or t < 6: traj.append(x[0].numpy().copy()); ts.append(t)
    traj = np.array(traj); tt = np.arange(D.N) / D.FS
    rng = np.random.default_rng(1); ref = np.array([synth_series(1.0, 1.0, 50.0, fs=D.FS, T=64.0, rng=rng)[0][:D.N] for _ in range(200)])
    fr, Pr = D.psd(ref); fig, ax = plt.subplots(1, 2, figsize=(12.8, 4.8), gridspec_kw=dict(width_ratios=[1.6, 1], wspace=.22))
    hold = 40; nfr = len(traj) + hold

    def upd(i):
        j = min(i, len(traj) - 1); ax[0].cla(); ax[1].cla()
        ax[0].plot(tt, traj[j], c=C["blue"], lw=1.1); ax[0].set(ylim=(-5, 8), xlim=(0, tt[-1]), xlabel="time  [s]", ylabel="u′ / σu", title=f"Reverse diffusion, step {ts[j]:>3d} / {D.TS}   (conditioned on a 5σ gust)")
        f, P = D.psd(traj[j][None]); ax[1].loglog(fr[1:], Pr[1:], c=INK2, ls="--", lw=1.4, label="Kaimal reference"); ax[1].loglog(f[1:], P[1:], c=C["orange"], label="current sample")
        ax[1].set(xlabel="frequency  [Hz]", ylabel="PSD", ylim=(1e-5, 1e2), title="Spectrum emerges from noise"); ax[1].legend(loc="lower left")
    fig.suptitle("Conditional diffusion model generating a turbulent wind series with an EOG gust", x=.01, ha="left", fontsize=14, fontweight="bold")
    render(fig, upd, nfr, "diffusion_wind_generation", fps=20, dpi=100)


def diurnal_video():
    from windlidar.cfd.lidar_validation import load_profiles, loglaw_fit
    import json
    z, t, W, Q = load_profiles(); hr = t.hour.values; sel = (z >= 60) & (z <= 150)
    prof = np.full((24, len(z)), np.nan); fit = []
    for h in range(24):
        m = hr == h
        prof[h] = [np.nanmedian(np.where(Q[m, k] == 1, W[m, k], np.nan)) if np.any(Q[m, k] == 1) else np.nan for k in range(len(z))]
        ok = sel & np.isfinite(prof[h]); fit.append(loglaw_fit(z[ok], prof[h][ok]) if ok.sum() >= 3 else (np.nan,) * 3)
    fit = np.array(fit); K = 0.4
    fig, ax = plt.subplots(1, 2, figsize=(12.8, 5.2), gridspec_kw=dict(width_ratios=[1.2, 1], wspace=.22))
    zz = np.geomspace(20, 450, 100)

    def upd(i):
        h = i % 24; ax[0].cla(); ax[1].cla()
        for hh in range(24): ax[0].plot(prof[hh], z, c=viz.GRID, lw=.8)
        night = h >= 19 or h <= 5; col = C["violet"] if night else C["blue"]
        ax[0].plot(prof[h], z, "o-", c=col, ms=4, lw=2, label=f"lidar median, {h:02d} UTC")
        us, z0, _ = fit[h]
        if np.isfinite(us) and z0 < 1e3: ax[0].plot(us / K * np.log(zz / z0), zz, c=C["orange"], ls="--", lw=1.6, label=f"log law: u* = {us:.2f} m/s, z0 = {z0:.3g} m")
        ax[0].axhspan(66, 146, color=C["aqua"], alpha=.08, lw=0); ax[0].text(.3, 100, "fit gates", color=INK2, fontsize=8)
        ax[0].set(ylim=(20, 450), xlim=(0, 11), yscale="log", xlabel="wind speed  [m/s]", ylabel="height  [m]", title="Median lidar wind profile per hour (FESSTVaL 2021)"); ax[0].legend(loc="upper left", fontsize=8.5)
        ax[1].semilogy(range(24), np.where(fit[:, 2] >= 0, fit[:, 1], np.nan), c=viz.AXIS, lw=1.2); ax[1].semilogy(range(h + 1), fit[:h + 1, 1], "o", c=C["blue"])
        ax[1].semilogy([h], [fit[h, 1]], "o", c=C["orange"], ms=10); ax[1].axhline(.1, c=MUTED, ls=":"); ax[1].text(.3, .13, "plausible farmland z0", fontsize=8.5, color=INK2)
        ax[1].axvspan(-1, 5.5, color=C["violet"], alpha=.08, lw=0); ax[1].axvspan(18.5, 24, color=C["violet"], alpha=.08, lw=0)
        ax[1].set(xlim=(-.5, 23.5), ylim=(1e-3, 30), xlabel="hour (UTC)", ylabel="fitted z0  [m]", title="Neutral log-law roughness")
        ax[1].text(.5, 12, "stable: neutral CFD not applicable" if night else "near-neutral: CFD valid", color=C["violet"] if night else C["blue"], fontsize=9.5)
    fig.suptitle("The daily cycle decides where the neutral RANS model is valid", x=.01, ha="left", fontsize=14, fontweight="bold")
    render(fig, upd, 24 * 2, "diurnal_profile_validity", fps=4, dpi=100)


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    for name, f in (("scan", scan_video), ("diffusion", diffusion_video), ("diurnal", diurnal_video)):
        if what in (name, "all"): f()
