"""Burst-signature feature windows + forward-looking event labels.

For each node and each time step, build a feature vector from the trailing
window W, and a label = does an event occur within the next horizon H.
"""
import numpy as np
import pandas as pd


def _burst_density(x, k=1.0):
    d = np.diff(x, prepend=x[0])
    thr = k * (np.std(d) + 1e-9)
    return (d > thr).astype(float)


def build(df_trace, df_sim, dt, cfg):
    W = max(1, int(cfg["features"]["window_seconds"] / dt))
    H = max(1, int(cfg["features"]["horizon_seconds"] / dt))
    # optional per-microservice columns (present only for the MSResource path)
    ms_cols = [c for c in ("n_active", "ms_std", "ms_max") if c in df_trace.columns]

    X, y, meta = [], [], []
    merged = df_trace.merge(df_sim, on=["node", "t_step"]).sort_values(["node", "t_step"])
    for node, g in merged.groupby("node"):
        g = g.reset_index(drop=True)
        cpu = g["cpu"].to_numpy(); mem = g["mem"].to_numpy()
        cur = g["current_a"].to_numpy(); call = g["callrate"].to_numpy()
        dcur = np.gradient(cur, dt)
        bd = _burst_density(cpu)
        event = g["event"].to_numpy().astype(int)
        ms = {c: g[c].to_numpy() for c in ms_cols}
        dms = {c: np.gradient(ms[c], dt) for c in ms_cols}   # trend of co-burst, etc.
        n = len(g)
        for i in range(W, n - H):
            wcpu = cpu[i - W:i]; wmem = mem[i - W:i]
            wcur = cur[i - W:i]; wd = dcur[i - W:i]; wbd = bd[i - W:i]; wcall = call[i - W:i]
            ac = np.corrcoef(wcur[:-1], wcur[1:])[0, 1] if W > 2 and np.std(wcur) > 0 else 0.0
            feat = [
                wcpu.mean(), wcpu.std(), np.percentile(wcpu, 95),
                wmem.mean(), wmem.std(),
                wcur.mean(), wcur.std(), wcur.max(),
                wd.mean(), wd.std(), wd.max(),
                wbd.sum(),
                wcall.mean(), np.nan_to_num(ac),
                cur[i - 1], dcur[i - 1],
            ]
            for c in ms_cols:                               # microservice burst signatures
                w = ms[c][i - W:i]
                feat += [w.mean(), w.std(), w.max(), ms[c][i - 1], dms[c][i - 1]]
            label = int(event[i:i + H].any())
            X.append(feat); y.append(label)
            meta.append((node, int(g["t_step"].iloc[i])))
    feat_names = ["cpu_mean","cpu_std","cpu_p95","mem_mean","mem_std",
                  "cur_mean","cur_std","cur_max","didt_mean","didt_std","didt_max",
                  "burst_density","call_mean","cur_autocorr","cur_last","didt_last"]
    for c in ms_cols:
        feat_names += [f"{c}_mean", f"{c}_std", f"{c}_max", f"{c}_last", f"{c}_delta"]
    return (np.asarray(X, float), np.asarray(y, int),
            pd.DataFrame(meta, columns=["node", "t_step"]), feat_names)
