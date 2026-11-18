"""Evaluation + reporting: Table I (event prediction), Table II (outcome),
Fig. 2 (waveforms), and results.json that the paper build reads to self-populate.
"""
import json, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (precision_score, recall_score, f1_score,
                             average_precision_score)

INK, BLUE, RED, GRID = "#1a1a1a", "#2166AC", "#B2182B", "#C9C9C9"


def temporal_split(meta, cfg):
    tr = np.zeros(len(meta), bool); va = np.zeros(len(meta), bool); te = np.zeros(len(meta), bool)
    vf, tf = cfg["model"]["val_fraction"], cfg["model"]["test_fraction"]
    for node, idx in meta.groupby("node").groups.items():
        idx = np.array(sorted(idx, key=lambda k: meta.loc[k, "t_step"]))
        n = len(idx); a = int(n * (1 - vf - tf)); b = int(n * (1 - tf))
        tr[idx[:a]] = True; va[idx[a:b]] = True; te[idx[b:]] = True
    return tr, va, te


def pick_threshold(scores, y, target_recall):
    order = np.argsort(-scores)
    ys = y[order]
    tp = np.cumsum(ys); total_pos = max(1, ys.sum())
    recall = tp / total_pos
    k = np.searchsorted(recall, target_recall)
    k = min(k, len(scores) - 1)
    return scores[order][k]


def _lead_ms(scores, thr, meta, df_sim, dt, H, test_mask):
    score_at = {(r.node, int(r.t_step)): s for r, s in zip(meta.itertuples(), scores)}
    test_steps = {(meta.loc[i, "node"], int(meta.loc[i, "t_step"])) for i in np.where(test_mask)[0]}
    leads = []
    for node, g in df_sim.sort_values("t_step").groupby("node"):
        ev = g["event"].to_numpy().astype(int); ts = g["t_step"].to_numpy()
        onsets = ts[1:][(ev[1:] == 1) & (ev[:-1] == 0)]
        for o in onsets:
            if (node, int(o)) not in test_steps:
                continue
            first = None
            for s in range(int(o) - H, int(o)):
                if score_at.get((node, s), -1e9) >= thr:
                    first = s; break
            if first is not None:
                leads.append((o - first) * dt * 1000.0)
    return float(np.median(leads)) if leads else 0.0


def evaluate(scores_dict, y, meta, df_sim, dt, cfg, backend):
    """Primary metric is PR-AUC (average precision) — threshold-free, the right
    metric for rare-event detection. Precision/recall/F1 are reported at a fixed
    alarm budget equal to the training-set base rate (flag the top-k riskiest
    windows), which transfers across time slices far better than a fixed score
    threshold. Lead time is measured at that operating point."""
    H = max(1, int(cfg["features"]["horizon_seconds"] / dt))
    tr, va, te = temporal_split(meta, cfg)
    base_rate = float(y[tr | va].mean()) or 0.1       # alarm budget (no test labels used)
    rows = []
    order = ["Reactive di/dt threshold", "Forecast + threshold", backend]
    yte = y[te]
    for name in order:
        sc = scores_dict[name]
        ste = sc[te]
        if yte.sum() == 0:
            rows.append(dict(model=name, pr_auc=float("nan"), precision=0.0,
                             recall=0.0, f1=0.0, lead_ms=0))
            continue
        ap = average_precision_score(yte, ste)
        k = max(1, int(round(base_rate * len(ste))))  # flag the k riskiest windows
        thr = np.sort(ste)[::-1][min(k - 1, len(ste) - 1)]
        pred = (ste >= thr).astype(int)
        p = precision_score(yte, pred, zero_division=0)
        r = recall_score(yte, pred, zero_division=0)
        f = f1_score(yte, pred, zero_division=0)
        lead = _lead_ms(sc, thr, meta, df_sim, dt, H, te)
        rows.append(dict(model=name, pr_auc=round(float(ap), 3), precision=round(p, 2),
                         recall=round(r, 2), f1=round(f, 2), lead_ms=round(lead)))
    return rows, (tr, va, te)


def render_fig2(fig, out_png):
    plt.rcParams.update({"font.family": "serif",
                         "font.serif": ["Liberation Serif", "Times New Roman"]})
    t, Ib, Ic, Vb, Vc, vf = fig["t"], fig["Ib"], fig["Ic"], fig["Vb"], fig["Vc"], fig["vfloor"]
    f, (a1, a2) = plt.subplots(2, 1, figsize=(3.4, 3.5), sharex=True,
                               gridspec_kw={"height_ratios": [1, 1.25], "hspace": 0.15})
    a1.plot(t, Ib, color=RED, lw=1.2, label="Baseline HPA")
    a1.plot(t, Ic, color=BLUE, lw=1.2, ls=(0, (5, 2)), label="Predictive (ours)")
    a1.set_ylabel("Node current\n$I_n(t)$ (A)", fontsize=7.5); a1.tick_params(labelsize=6.5)
    a1.grid(True, color=GRID, lw=0.5, alpha=0.6)
    a1.legend(fontsize=6.0, frameon=False, loc="upper right", ncol=2, columnspacing=1.0)
    for s in ("top", "right"): a1.spines[s].set_visible(False)
    a2.axhline(vf, color=INK, lw=0.9, ls=":")
    a2.text(t[1], vf + 1, "$V_{min}$ + margin", fontsize=6.0, va="bottom")
    a2.fill_between(t, Vb, vf, where=(Vb < vf), color=RED, alpha=0.18)
    a2.plot(t, Vb, color=RED, lw=1.2); a2.plot(t, Vc, color=BLUE, lw=1.2, ls=(0, (5, 2)))
    a2.set_ylabel("Die voltage\n$V_n(t)$ (mV)", fontsize=7.5)
    a2.set_xlabel("Time (s)", fontsize=7.5); a2.tick_params(labelsize=6.5)
    a2.grid(True, color=GRID, lw=0.5, alpha=0.6)
    for s in ("top", "right"): a2.spines[s].set_visible(False)
    f.subplots_adjust(left=0.19, right=0.985, top=0.985, bottom=0.11)
    f.savefig(out_png, dpi=300); plt.close(f)


def write_results(out_dir, table1, table2, backend, cfg, fig):
    os.makedirs(out_dir, exist_ok=True)
    fig_png = os.path.join(out_dir, "fig2_droop.png")
    render_fig2(fig, fig_png)
    results = {
        "provenance": cfg["controller"]["provenance"],
        "dataset": cfg["data"]["mode"],
        "backend": backend,
        "pdn": {k: cfg["pdn"][k] for k in ("r_ohm", "l_henry", "vmin_mv", "vnom_mv")},
        "table1": table1,
        "table2": table2,
        "fig2_png": os.path.abspath(fig_png),
    }
    with open(os.path.join(out_dir, "results.json"), "w") as fh:
        json.dump(results, fh, indent=2)
    # human-readable tables
    with open(os.path.join(out_dir, "tables.md"), "w") as fh:
        fh.write(f"Provenance: {results['provenance']}  |  backend: {backend}\n\n")
        fh.write("Table I — event prediction (PR-AUC is the headline metric)\n")
        fh.write("model | PR-AUC | precision | recall | f1 | lead(ms)\n")
        for r in table1:
            fh.write(f"{r['model']} | {r.get('pr_auc')} | {r['precision']} | {r['recall']} | {r['f1']} | {r['lead_ms']}\n")
        fh.write("\nTable II — power-integrity outcome\n")
        for r in table2:
            fh.write(" | ".join(r) + "\n")
    return results
