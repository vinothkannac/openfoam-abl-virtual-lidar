"""Small MLP and helpers to save and load a trained predictor (weights, input/output scaling, feature list).

I chose a small MLP (about 10^3-10^4 parameters) because the mapping has only two or three inputs and a smooth shape; anything larger only fits
the noise of 10-minute gust maxima. The feature sets reflect what each data product offers: "full" needs the lidar turbulence intensity, "level2"
only mean wind and gust (what the public FESSTVaL level-2 files contain), "level2_hour" adds hour of day as a stability proxy.
"""
import numpy as np
import torch, torch.nn as nn

FEATURES = {
    "full":   ["U_lidar", "TI_lidar", "GF_lidar"],   # needs the lidar 10-min mean, turbulence intensity and gust factor
    "level2": ["U_lidar", "GF_lidar"],               # works with FESSTVaL level-2 products (mean wind + gust only)
    "level2_hour": ["U_lidar", "GF_lidar", "sin_h", "cos_h"],   # + hour of day (UTC) as a stability proxy
}

class MLP(nn.Module):
    def __init__(self, n_in, hidden, act="relu", n_out=1):
        super().__init__()
        A = {"relu": nn.ReLU, "gelu": nn.GELU, "tanh": nn.Tanh}[act]
        layers, d = [], n_in
        for h in hidden:
            layers += [nn.Linear(d, h), A()]; d = h
        layers.append(nn.Linear(d, n_out))
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x)

class Predictor:
    def __init__(self, model, mu_x, sd_x, mu_y, sd_y, features, target, cfg):
        self.model, self.mu_x, self.sd_x, self.mu_y, self.sd_y = model.eval(), mu_x, sd_x, mu_y, sd_y
        self.features, self.target, self.cfg = features, target, cfg
    @torch.no_grad()
    def predict(self, X):
        X = np.atleast_2d(np.asarray(X, dtype=np.float32))
        z = torch.from_numpy((X - self.mu_x) / self.sd_x)
        return self.model(z).numpy().ravel() * self.sd_y + self.mu_y

def save_predictor(path, model, mu_x, sd_x, mu_y, sd_y, features, target, cfg):
    torch.save(dict(state=model.state_dict(), mu_x=mu_x, sd_x=sd_x, mu_y=float(mu_y), sd_y=float(sd_y),
                    features=features, target=target, cfg=cfg), path)

def load_predictor(path):
    d = torch.load(path, map_location="cpu", weights_only=False)
    cfg = d["cfg"]
    m = MLP(len(d["features"]), cfg["hidden"], cfg["act"]); m.load_state_dict(d["state"])
    return Predictor(m, d["mu_x"], d["sd_x"], d["mu_y"], d["sd_y"], d["features"], d["target"], cfg)

def split_indices(df, seed=0):
    """Reproducible split used by train.py and uncertainty.py.  'unseen' = L_eff > 130 m or u* > 0.8 m/s (never trained on)."""
    rng = np.random.default_rng(seed)
    ood = ((df.L_eff > 130) | (df.ustar > 0.8)).values
    idx = np.where(~ood)[0]; rng.shuffle(idx); n = len(idx)
    return idx[:int(.7 * n)], idx[int(.7 * n):int(.85 * n)], idx[int(.85 * n):], np.where(ood)[0]
