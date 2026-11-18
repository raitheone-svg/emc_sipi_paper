"""Trace-to-droop co-simulation: utilization -> power -> PDN voltage + RC thermal
-> droop / throttle event labels. This is the transparent, swept 'translator'
from the paper; it is the ONE place the electrical assumptions live.
"""
import numpy as np
import pandas as pd


def node_power(cpu, mem, cfg):
    p = cfg["power"]
    span = p["p_peak_w"] - p["p_idle_w"]
    # dynamic power tracks total (summed) container activity; no upper saturation
    # so node-current variation is preserved for the PDN response.
    util = np.clip(cpu, 0, None) + p["mem_weight"] * np.clip(mem, 0, None)
    return p["p_idle_w"] + span * util


def pdn_voltage(current, dt, cfg):
    """V(t) = Vnom - R*(I-I0) - L*dI/dt. di/dt from the discrete first difference
    at the trace resolution (slow-droop / envelope regime; see paper Sec. VIII)."""
    pdn = cfg["pdn"]
    I0 = np.median(current)
    didt = np.gradient(current, dt)
    v = pdn["vnom_mv"] - (pdn["r_ohm"] * (current - I0)) * 1000.0 \
        - (pdn["l_henry"] * didt) * 1000.0
    return v


def junction_temp(power, dt, cfg):
    """First-order RC thermal step response."""
    th = cfg["thermal"]
    alpha = dt / (th["tau_seconds"] + dt)
    t = np.empty_like(power)
    t[0] = th["t_ambient_c"] + th["r_th_c_per_w"] * (power[0] - cfg["power"]["p_idle_w"])
    target = th["t_ambient_c"] + th["r_th_c_per_w"] * (power - cfg["power"]["p_idle_w"])
    for i in range(1, len(power)):
        t[i] = t[i - 1] + alpha * (target[i] - t[i - 1])
    return t


def simulate(df, dt, cfg):
    """Returns a per-(node,t_step) frame with current, voltage, temp, and
    boolean droop/throttle/event labels.

    Voltage floor: if pdn.target_event_rate is set (e.g. 0.10), the guardband is
    placed at the workload's own droop distribution so that fraction of the
    deepest droops violate it (the aggressive-margining regime). This keeps the
    event rate stable across datasets. Otherwise the fixed pdn.vmin_mv is used.
    The resolved floor is written to cfg['pdn']['_resolved_vfloor_mv'] so the
    controller and report use the SAME threshold."""
    pdn = cfg["pdn"]
    tlimit = cfg["thermal"]["t_limit_c"]

    # pass 1: compute electrical response per node
    recs = []
    for node, g in df.sort_values("t_step").groupby("node"):
        cpu = g["cpu"].to_numpy(); mem = g["mem"].to_numpy()
        P = node_power(cpu, mem, cfg)
        I = P / cfg["power"]["vdd_v"]
        V = pdn_voltage(I, dt, cfg)
        T = junction_temp(P, dt, cfg)
        o = g[["node", "t_step"]].copy()
        o["power_w"] = P; o["current_a"] = I; o["voltage_mv"] = V; o["temp_c"] = T
        recs.append(o)
    allrec = pd.concat(recs, ignore_index=True)

    # resolve the voltage floor
    target = pdn.get("target_event_rate")
    if target:
        vfloor = float(np.quantile(allrec["voltage_mv"].to_numpy(), target))
    else:
        vfloor = pdn["vmin_mv"] + pdn["margin_mv"]
    cfg["pdn"]["_resolved_vfloor_mv"] = vfloor

    # pass 2: label. With a target_event_rate the event = droop only (so the
    # rate is exactly the target); otherwise droop OR thermal throttle.
    allrec["droop"] = allrec["voltage_mv"] < vfloor
    allrec["throttle"] = allrec["temp_c"] > tlimit
    allrec["event"] = allrec["droop"] if target else (allrec["droop"] | allrec["throttle"])
    return allrec
