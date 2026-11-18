#!/usr/bin/env python3
"""End-to-end runner: trace -> co-sim labels -> features -> models -> controller
-> results.json + Fig. 2. Runs anywhere in synthetic mode; point config.data.mode
at 'alibaba' or 'google2011' (with data downloaded) for the real experiment.

Usage:
    python run_all.py --config config.yaml
"""
import argparse, yaml
import numpy as np
from src import io_trace, cosim, features, model, controller, report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    rng = np.random.default_rng(cfg["run"]["seed"])

    print(f"[1/6] loading trace (mode={cfg['data']['mode']}) ...")
    df_trace, dt = io_trace.load(cfg, rng)
    print(f"      nodes={df_trace['node'].nunique()}  rows={len(df_trace)}  dt={dt}s")

    print("[2/6] co-simulation (power -> PDN -> thermal -> labels) ...")
    df_sim = cosim.simulate(df_trace, dt, cfg)
    ev_rate = df_sim["event"].mean()
    print(f"      event rate = {ev_rate:.3f}  (droop {df_sim['droop'].mean():.3f}, "
          f"throttle {df_sim['throttle'].mean():.3f})")

    print("[3/6] feature windows + labels ...")
    X, y, meta, names = features.build(df_trace, df_sim, dt, cfg)
    print(f"      samples={len(X)}  positives={int(y.sum())}  dim={X.shape[1]}")

    print("[4/6] training predictor + baselines ...")
    H = max(1, int(cfg["features"]["horizon_seconds"] / dt))
    tr, va, te = report.temporal_split(meta, cfg)
    scores, backend = model.train_all(X, y, names, tr, va, H, cfg, rng)
    print(f"      learned backend = {backend}")

    print("[5/6] evaluating (Table I) ...")
    table1, _ = report.evaluate(scores, y, meta, df_sim, dt, cfg, backend)
    for r in table1:
        print(f"      {r['model']:<32} P={r['precision']} R={r['recall']} "
              f"F1={r['f1']} lead={r['lead_ms']}ms")

    print("[6/6] controller replay + outcome (Table II) ...")
    table2, fig = controller.run(df_trace, df_sim, dt, scores[backend], meta, cfg)
    for r in table2:
        print("      " + " | ".join(r))

    res = report.write_results(cfg["run"]["out_dir"], table1, table2, backend, cfg, fig)
    print(f"\nWrote {cfg['run']['out_dir']}/results.json  and  fig2_droop.png")
    print(f"Provenance: {res['provenance']}")


if __name__ == "__main__":
    main()
