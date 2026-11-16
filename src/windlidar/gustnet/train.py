"""Train and compare four MLP configurations that correct the lidar gust factor towards the sonic reference.

Choices: four small architectures (ReLU/GELU, different widths) so that model selection is on validation loss and not by hand; a random
70/15/15 split plus an extra "unseen" test set (L_eff > 130 m or u* > 0.8 m/s) that is never used for training or selection, to test
extrapolation instead of interpolation. The dataset is small, so a GPU gives no real speed-up (--device auto uses one if present).
"""
import argparse, json, os, shutil, time
from windlidar.paths import rp
import numpy as np, pandas as pd, torch, torch.nn as nn
from windlidar.gustnet.model import MLP, FEATURES, save_predictor, split_indices

CONFIGS = [
    dict(name="A_small_relu",  hidden=[64, 64],            act="relu", lr=1e-3, wd=0.0),
    dict(name="B_medium_relu", hidden=[128, 128, 64],      act="relu", lr=1e-3, wd=0.0),
    dict(name="C_medium_gelu", hidden=[128, 128, 64],      act="gelu", lr=1e-3, wd=1e-5),
    dict(name="D_deep_relu",   hidden=[256, 256, 128, 64], act="relu", lr=5e-4, wd=1e-5),
]

def r2(y, p): return float(1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2))
def mae(y, p): return float(np.mean(np.abs(y - p)))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=rp("data/dataset.csv"))
    ap.add_argument("--features", default="full", choices=list(FEATURES))
    ap.add_argument("--target", default="GF_sonic")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--patience", type=int, default=30)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default=rp("models/baseline"))
    ap.add_argument("--results", default=rp("results/gustnet"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip-done", action="store_true", help="skip configs whose checkpoint already exists (Colab restarts)")
    a = ap.parse_args()
    dev = torch.device("cuda" if (a.device == "auto" and torch.cuda.is_available()) else ("cpu" if a.device == "auto" else a.device))
    print("device:", dev, "| torch", torch.__version__)
    torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)
    os.makedirs(a.out, exist_ok=True); os.makedirs(a.results, exist_ok=True)

    df = pd.read_csv(a.data); feats = FEATURES[a.features]
    tr, va, te, oo = split_indices(df, a.seed)
    X, y = df[feats].values.astype(np.float32), df[a.target].values.astype(np.float32)
    mu_x, sd_x = X[tr].mean(0), X[tr].std(0) + 1e-8; mu_y, sd_y = y[tr].mean(), y[tr].std() + 1e-8
    print("train/val/test/unseen:", len(tr), len(va), len(te), len(oo))

    # ---- baselines: (1) raw lidar value, (2) linear regression on the same features
    raw = df["GF_lidar"].values if a.target == "GF_sonic" else df["TI_lidar"].values
    A = np.c_[X[tr], np.ones(len(tr))]; coef = np.linalg.lstsq(A, y[tr], rcond=None)[0]
    lin = lambda ii: np.c_[X[ii], np.ones(len(ii))] @ coef
    summary = {"baseline_raw_lidar": {"test": dict(R2=r2(y[te], raw[te]), MAE=mae(y[te], raw[te])),
                                      "unseen": dict(R2=r2(y[oo], raw[oo]), MAE=mae(y[oo], raw[oo]))},
               "baseline_linear": {"test": dict(R2=r2(y[te], lin(te)), MAE=mae(y[te], lin(te))),
                                   "unseen": dict(R2=r2(y[oo], lin(oo)), MAE=mae(y[oo], lin(oo)))}}
    print(json.dumps(summary, indent=1))

    T = lambda ii: (torch.from_numpy((X[ii] - mu_x) / sd_x).to(dev), torch.from_numpy(((y[ii] - mu_y) / sd_y)[:, None]).to(dev))
    Xtr, ytr = T(tr); Xva, yva = T(va)
    hist_all, results = {}, {}
    for cfg in CONFIGS:
        ck = os.path.join(a.out, "cfg_%s.pt" % cfg["name"])
        model = MLP(len(feats), cfg["hidden"], cfg["act"]).to(dev)
        if a.skip_done and os.path.exists(ck):
            d = torch.load(ck, map_location=dev, weights_only=False); model.load_state_dict(d["state"])
            hist = json.load(open(os.path.join(a.out, "hist_%s.json" % cfg["name"]))); print("loaded", cfg["name"])
        else:
            opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
            sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=10)
            best, best_state, bad, hist, t0 = 1e9, None, 0, dict(train=[], val=[]), time.time()
            for ep in range(a.epochs):
                model.train(); perm = torch.randperm(len(Xtr), device=dev); tl = 0.0
                for i in range(0, len(perm), a.batch):
                    b = perm[i:i + a.batch]; opt.zero_grad()
                    loss = nn.functional.mse_loss(model(Xtr[b]), ytr[b]); loss.backward(); opt.step(); tl += loss.item() * len(b)
                model.eval()
                with torch.no_grad(): vl = nn.functional.mse_loss(model(Xva), yva).item()
                hist["train"].append(tl / len(Xtr)); hist["val"].append(vl); sch.step(vl)
                if vl < best - 1e-7: best, bad, best_state = vl, 0, {k: v.clone() for k, v in model.state_dict().items()}
                else: bad += 1
                if bad >= a.patience: break
            model.load_state_dict(best_state)
            print("%s: %d epochs, %.1f s, best val MSE (scaled) %.4f" % (cfg["name"], ep + 1, time.time() - t0, best))
            torch.save(dict(state=model.state_dict()), ck); json.dump(hist, open(os.path.join(a.out, "hist_%s.json" % cfg["name"]), "w"))
        model.eval()
        pred = lambda ii: model(torch.from_numpy((X[ii] - mu_x) / sd_x).to(dev)).detach().cpu().numpy().ravel() * sd_y + mu_y
        results[cfg["name"]] = dict(val=dict(R2=r2(y[va], pred(va)), MAE=mae(y[va], pred(va))),
                                    test=dict(R2=r2(y[te], pred(te)), MAE=mae(y[te], pred(te))),
                                    unseen=dict(R2=r2(y[oo], pred(oo)), MAE=mae(y[oo], pred(oo))), cfg=cfg)
        hist_all[cfg["name"]] = hist
        print(cfg["name"], {k: results[cfg["name"]][k] for k in ("val", "test", "unseen")})
    best_name = min(results, key=lambda k: results[k]["val"]["MAE"])
    bc = results[best_name]["cfg"]
    m = MLP(len(feats), bc["hidden"], bc["act"]); m.load_state_dict(torch.load(os.path.join(a.out, "cfg_%s.pt" % best_name), map_location="cpu", weights_only=False)["state"])
    save_predictor(os.path.join(a.out, "best_%s.pt" % a.features), m, mu_x, sd_x, mu_y, sd_y, feats, a.target, bc)
    # residual std on validation (used by uncertainty.py)
    with torch.no_grad(): pv = m(torch.from_numpy((X[va] - mu_x) / sd_x)).numpy().ravel() * sd_y + mu_y
    summary.update(models=results, best=best_name, features=feats, target=a.target, val_residual_std=float(np.std(y[va] - pv)),
                   n=dict(train=len(tr), val=len(va), test=len(te), unseen=len(oo)), device=str(dev))
    # inference speed (CPU)
    t0 = time.time(); _ = m(torch.from_numpy((X[te] - mu_x) / sd_x)); summary["inference_us_per_sample"] = 1e6 * (time.time() - t0) / len(te)
    json.dump(summary, open(os.path.join(a.results, "metrics_%s.json" % a.features), "w"), indent=1)
    print("best:", best_name); print(json.dumps({k: summary[k] for k in ("baseline_raw_lidar", "baseline_linear")}, indent=1))

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.3))
    for k, h in hist_all.items(): ax[0].plot(h["val"], label=k)
    ax[0].set_yscale("log"); ax[0].set_xlabel("epoch"); ax[0].set_ylabel("validation MSE (scaled)"); ax[0].legend(fontsize=7); ax[0].set_title("Learning curves")
    p = m(torch.from_numpy((X[te] - mu_x) / sd_x)).detach().numpy().ravel() * sd_y + mu_y
    ax[1].scatter(y[te], raw[te], s=4, alpha=.3, label="raw lidar"); ax[1].scatter(y[te], p, s=4, alpha=.3, label="corrected (%s)" % best_name)
    lo, hi = y.min(), y.max(); ax[1].plot([lo, hi], [lo, hi], "k--", lw=1); ax[1].set_xlabel("sonic %s" % a.target); ax[1].set_ylabel("lidar / corrected"); ax[1].legend(); ax[1].set_title("Test set")
    po = m(torch.from_numpy((X[oo] - mu_x) / sd_x)).detach().numpy().ravel() * sd_y + mu_y
    ax[2].scatter(df.L_eff[oo], po - y[oo], s=4, alpha=.4, label="unseen: error after correction"); ax[2].scatter(df.L_eff[te], p - y[te], s=4, alpha=.2, label="test")
    ax[2].axhline(0, c="k", lw=1); ax[2].set_xlabel("latent L_eff [m]"); ax[2].set_ylabel("corrected - sonic"); ax[2].legend(); ax[2].set_title("Error vs latent lidar length")
    fig.tight_layout(); fig.savefig(os.path.join(a.results, "training_%s.png" % a.features), dpi=130)

if __name__ == "__main__":
    main()
