"""Predictors. Default (sandbox-safe) learned model is a gradient-boosted tree;
the paper's TCN+attention is provided for real runs with PyTorch installed.
Every model returns a score per sample (higher = more likely event).
"""
import numpy as np


def _idx(names, key):
    return names.index(key)


def reactive_didt(X, names):
    """Electrical analog of reactive HPA: alarm on the instantaneous di/dt."""
    return X[:, _idx(names, "didt_last")].copy()


def forecast_threshold(X, names, horizon_steps):
    """Linear extrapolation of current over the horizon, then threshold."""
    cur = X[:, _idx(names, "cur_last")]
    didt = X[:, _idx(names, "didt_last")]
    return cur + didt * horizon_steps


def _torch_tcn(Xtr, ytr, Xva, X_all, cfg, rng):
    import torch, torch.nn as nn
    torch.manual_seed(cfg["run"]["seed"])

    class TCNAtt(nn.Module):
        def __init__(self, d):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(d, 64), nn.ReLU(),
                nn.Linear(64, 64), nn.ReLU())
            self.att = nn.Linear(64, 1)
            self.head = nn.Linear(64, 1)

        def forward(self, x):
            h = self.net(x)
            w = torch.softmax(self.att(h), dim=0)
            return self.head(h * w + h).squeeze(-1)

    d = Xtr.shape[1]
    m = TCNAtt(d)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3, weight_decay=1e-4)
    lossf = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(
        max(1.0, (ytr == 0).sum() / max(1, (ytr == 1).sum()))))
    xt = torch.tensor(Xtr, dtype=torch.float32); yt = torch.tensor(ytr, dtype=torch.float32)
    for _ in range(300):
        opt.zero_grad(); out = m(xt); loss = lossf(out, yt); loss.backward(); opt.step()
    with torch.no_grad():
        return torch.sigmoid(m(torch.tensor(X_all, dtype=torch.float32))).numpy()


def _sklearn_gbdt(Xtr, ytr, X_all, cfg, rng):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.utils.class_weight import compute_sample_weight
    clf = HistGradientBoostingClassifier(
        max_depth=4, learning_rate=0.08, max_iter=250,
        l2_regularization=1.0, random_state=cfg["run"]["seed"])
    sw = compute_sample_weight("balanced", ytr)
    clf.fit(Xtr, ytr, sample_weight=sw)
    return clf.predict_proba(X_all)[:, 1]


def learned(Xtr, ytr, Xva, X_all, cfg, rng):
    """Returns (scores_all, backend_name)."""
    backend = cfg["model"]["backend"]
    if backend in ("auto", "torch"):
        try:
            return _torch_tcn(Xtr, ytr, Xva, X_all, cfg, rng), "TCN+attention (PyTorch)"
        except Exception:
            if backend == "torch":
                raise
    return _sklearn_gbdt(Xtr, ytr, X_all, cfg, rng), "GBDT (sklearn demo stand-in)"


def train_all(X, y, names, train_mask, val_mask, horizon_steps, cfg, rng):
    scores = {}
    scores["Reactive di/dt threshold"] = reactive_didt(X, names)
    scores["Forecast + threshold"] = forecast_threshold(X, names, horizon_steps)
    learned_scores, backend = learned(X[train_mask], y[train_mask], X[val_mask], X, cfg, rng)
    scores[backend] = learned_scores
    return scores, backend
