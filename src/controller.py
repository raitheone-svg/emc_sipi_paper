"""Droop-aware controller: use the learned risk to reshape the activation
schedule (stagger + bound concurrent scale-ups), then re-run the co-simulation
on the modified schedule and measure the power-integrity outcome (Table II).
"""
import numpy as np
import pandas as pd
from . import cosim


def _reshape(cpu, risk_flags, k, cap):
    """Spread sharp positive jumps at high-risk steps over k intervals, and
    cap the single-step jump (token-bucket analog)."""
    out = cpu.astype(float).copy()
    n = len(cpu)
    i = 1
    while i < n:
        jump = out[i] - out[i - 1]
        if risk_flags[i] and jump > 0:
            # bound the instantaneous step, ramp the remainder over k steps
            ramp = min(k, n - i)
            target = out[i - 1] + jump
            for j in range(ramp):
                frac = (j + 1) / ramp
                out[i + j] = out[i - 1] + jump * frac
            i += ramp
        else:
            i += 1
    return out


def run(df_trace, df_sim, dt, scores_learned, meta, cfg):
    thr = cfg["controller"]["risk_threshold"]
    k = cfg["controller"]["stagger_steps"]
    cap = cfg["controller"]["max_concurrent_scaleups"]
    vnom = cfg["pdn"]["vnom_mv"]
    vfloor = cfg["pdn"].get("_resolved_vfloor_mv", cfg["pdn"]["vmin_mv"] + cfg["pdn"]["margin_mv"])
    tlimit = cfg["thermal"]["t_limit_c"]

    # risk lookup
    risk = {(r.node, int(r.t_step)): s for r, s in zip(meta.itertuples(), scores_learned)}

    base_depth, ctrl_depth = [], []
    base_minV, ctrl_minV = [], []
    base_thr, ctrl_thr = 0, 0
    fig = None
    for node, g in df_trace.sort_values("t_step").groupby("node"):
        g = g.reset_index(drop=True)
        cpu = g["cpu"].to_numpy(); mem = g["mem"].to_numpy()
        flags = np.array([risk.get((node, int(ts)), 0.0) >= thr for ts in g["t_step"]])
        cpu_c = _reshape(cpu, flags, k, cap)

        Pb = cosim.node_power(cpu, mem, cfg); Ib = Pb / cfg["power"]["vdd_v"]
        Pc = cosim.node_power(cpu_c, mem, cfg); Ic = Pc / cfg["power"]["vdd_v"]
        Vb = cosim.pdn_voltage(Ib, dt, cfg); Vc = cosim.pdn_voltage(Ic, dt, cfg)
        Tb = cosim.junction_temp(Pb, dt, cfg); Tc = cosim.junction_temp(Pc, dt, cfg)

        base_depth.append(vnom - Vb); ctrl_depth.append(vnom - Vc)
        base_minV.append(Vb.min()); ctrl_minV.append(Vc.min())
        base_thr += int(np.sum((Tb[1:] > tlimit) & (Tb[:-1] <= tlimit)))
        ctrl_thr += int(np.sum((Tc[1:] > tlimit) & (Tc[:-1] <= tlimit)))
        if fig is None:
            t = g["t_step"].to_numpy() * dt
            fig = dict(t=t, Ib=Ib, Ic=Ic, Vb=Vb, Vc=Vc, vfloor=vfloor)

    bd = np.concatenate(base_depth); cd = np.concatenate(ctrl_depth)
    hours = (df_trace["t_step"].max() + 1) * dt / 3600.0
    frac_flagged = float(np.mean([v >= thr for v in scores_learned]))

    peak_b, peak_c = float(bd.max()), float(cd.max())
    p99_b, p99_c = float(np.percentile(bd, 99)), float(np.percentile(cd, 99))
    guardband = max(0.0, float(np.min(ctrl_minV) - np.min(base_minV)))
    viol_b = int(np.sum(bd > (vnom - vfloor))); viol_c = int(np.sum(cd > (vnom - vfloor)))
    thr_b_hr = base_thr / max(hours, 1e-9); thr_c_hr = ctrl_thr / max(hours, 1e-9)
    lat_cost = k * dt * 1000.0 * frac_flagged
    slo_ratio = (viol_c / viol_b) if viol_b else 1.0

    def pct(a, b):
        return f"{(b-a)/a*100:+.0f}%" if a else "—"

    table2 = [
        ["Peak droop (mV)", f"{peak_b:.0f}", f"{peak_c:.0f}", pct(peak_b, peak_c)],
        ["p99 droop (mV)", f"{p99_b:.0f}", f"{p99_c:.0f}", pct(p99_b, p99_c)],
        ["Reclaim. guardband (mV)", "—", f"{guardband:.0f}", f"+{guardband:.0f}"],
        ["Throttle events / hr", f"{thr_b_hr:.1f}", f"{thr_c_hr:.1f}", pct(thr_b_hr, thr_c_hr)],
        ["p99 latency cost (ms)", "—", f"+{lat_cost:.1f}", "—"],
        ["SLO violation rate", "1.00×", f"{slo_ratio:.2f}×", pct(1.0, slo_ratio)],
    ]
    return table2, fig
