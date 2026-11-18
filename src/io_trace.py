"""Trace loaders -> a common tidy node-level frame.

Common output: a pandas DataFrame indexed by (node, t_step) with columns
    cpu   : node aggregate CPU utilization proxy in [0, ~1+]
    mem   : node aggregate memory utilization proxy in [0, ~1]
    callrate (optional), fanout (optional)
Every downstream stage consumes ONLY this frame, so synthetic / Alibaba /
Google data are interchangeable.
"""
import numpy as np
import pandas as pd


# ---------------------------------------------------------------- synthetic
def make_synthetic(cfg, rng):
    d = cfg["data"]
    dt = d["dt_seconds"]
    n_steps = int(d["duration_seconds"] / dt)
    t = np.arange(n_steps) * dt
    rows = []
    # shared "orchestrator triggers": correlated activation instants
    n_triggers = max(4, int(d["duration_seconds"] / 40))
    trig_times = np.sort(rng.uniform(t[0] + 5, t[-1] - 5, n_triggers))
    for node in range(d["n_nodes"]):
        cpu = np.full(n_steps, 0.0)
        mem = np.full(n_steps, 0.0)
        for s in range(d["services_per_node"]):
            base = rng.uniform(0.01, 0.05)
            svc = np.full(n_steps, base)
            # idiosyncratic small bursts
            for _ in range(rng.integers(3, 8)):
                c = rng.uniform(t[0], t[-1]); w = rng.uniform(0.3, 1.2)
                svc += rng.uniform(0.05, 0.2) * np.exp(-((t - c) ** 2) / (2 * w ** 2))
            # correlated bursts: most services on a node react to shared triggers
            if rng.random() < 0.6:
                for c in trig_times:
                    jitter = rng.normal(0, 0.15)
                    w = rng.uniform(0.15, 0.4)   # sharp, near-simultaneous
                    svc += rng.uniform(0.15, 0.5) * np.exp(-((t - (c + jitter)) ** 2) / (2 * w ** 2))
            cpu += svc
            mem += 0.4 * svc + base
        cpu = cpu / d["services_per_node"] * 8.0           # scale to a busy node
        mem = np.clip(mem / d["services_per_node"] * 5.0, 0, 1.2)
        callrate = cpu * rng.uniform(80, 120)
        for i in range(n_steps):
            rows.append((node, i, float(cpu[i]), float(mem[i]), float(callrate[i]), 0.0))
    df = pd.DataFrame(rows, columns=["node", "t_step", "cpu", "mem", "callrate", "fanout"])
    return df, dt


# ---------------------------------------------------------------- alibaba
def load_alibaba(cfg):
    """Loader skeleton for the Alibaba microservice trace (v2021/v2022).

    Expects pre-downloaded CSV/parquet parts in cfg['data']['data_dir']:
      MSResource / MSRTMCR : per-microservice CPU, memory, call rate
      MSCallGraph          : caller -> callee edges (for fan-out)
      Node                 : microservice -> node (pod -> machine) mapping
    Fill the column names to match the release you downloaded; the README
    lists the exact fields. Returns the same tidy frame as make_synthetic.
    """
    import os, glob
    dd = cfg["data"]["data_dir"]
    dt = cfg["data"]["dt_seconds"]
    max_nodes = int(cfg["data"].get("max_nodes", 50))

    NODE_COLS = ["timestamp", "nodeid", "cpu_utilization", "memory_utilization"]
    MSRES_COLS = ["timestamp", "msname", "msinstanceid", "nodeid",
                  "cpu_utilization", "memory_utilization"]
    alias = {"cpu_utilization": "cpu", "memory_utilization": "mem", "node_id": "nodeid"}
    CHUNK = 1_000_000

    def _find(*names, exclude=()):
        out = []
        for name in names:
            out += glob.glob(os.path.join(dd, "**", f"*{name}*.csv"), recursive=True)
            out += glob.glob(os.path.join(dd, "**", f"*{name}*.CSV"), recursive=True)
        return [f for f in sorted(set(out)) if os.path.isfile(f)
                and not any(x in os.path.basename(f) for x in exclude)]

    def _detect(path):
        head = pd.read_csv(path, nrows=1, header=None)
        v = head.iloc[0, 0]
        has_header = isinstance(v, str) and not str(v).replace(".", "", 1).isdigit()
        return has_header, head.shape[1]

    def _iter(path, schema):
        """Stream a (possibly huge) CSV in chunks, header or header-less,
        with columns renamed to cpu/mem/nodeid/timestamp."""
        has_header, ncols = _detect(path)
        if has_header:
            header, names = 0, None
        else:
            names = (schema + [f"extra{i}" for i in range(max(0, ncols - len(schema)))])[:ncols]
            header = None
        for chunk in pd.read_csv(path, header=header, names=names, chunksize=CHUNK):
            yield chunk.rename(columns=alias)

    # ---- choose table: MSResource (per-container, RICH) preferred ------
    ACTIVE_THR = 0.05
    ms_files = _find("MSMetrics", "MSResource",
                     exclude=("MSRTMCR", "CallGraph", "NodeMetrics", "Node"))
    node_files = _find("NodeMetrics", "Node",
                       exclude=("MSMetrics", "MSResource", "MSRTMCR", "CallGraph"))

    if ms_files:
        # pass 1: approximate busiest nodes by per-chunk node CPU sum
        peak = {}
        for p in ms_files:
            for ch in _iter(p, MSRES_COLS):
                ch = ch.assign(t_step=(ch["timestamp"] / 1000.0 / dt).astype(int))
                s = ch.groupby(["nodeid", "t_step"])["cpu"].sum().groupby("nodeid").max()
                for nid, v in s.items():
                    if v > peak.get(nid, -1.0):
                        peak[nid] = v
        top = set(sorted(peak, key=peak.get, reverse=True)[:max_nodes])

        # pass 2: per (node,t_step) accumulate container stats (chunk-safe)
        parts = []
        for p in ms_files:
            for ch in _iter(p, MSRES_COLS):
                ch = ch[ch["nodeid"].isin(top)]
                if not len(ch):
                    continue
                ch = ch.assign(t_step=(ch["timestamp"] / 1000.0 / dt).astype(int),
                               cpu2=lambda d: d["cpu"] ** 2,
                               act=lambda d: (d["cpu"] > ACTIVE_THR).astype(float))
                parts.append(ch.groupby(["nodeid", "t_step"]).agg(
                    cnt=("cpu", "size"), s=("cpu", "sum"), sq=("cpu2", "sum"),
                    mx=("cpu", "max"), nact=("act", "sum"),
                    msum=("mem", "sum"), mcnt=("mem", "size")).reset_index())
        a = (pd.concat(parts, ignore_index=True).groupby(["nodeid", "t_step"])
               .agg(cnt=("cnt", "sum"), s=("s", "sum"), sq=("sq", "sum"),
                    mx=("mx", "max"), nact=("nact", "sum"),
                    msum=("msum", "sum"), mcnt=("mcnt", "sum")).reset_index())
        a["cpu"] = a["s"]                                   # node current proxy
        scale = a["cpu"].quantile(0.95)                     # normalize so p95 ~ 1
        if scale and scale > 0:
            a["cpu"] = a["cpu"] / scale
        a["mem"] = a["msum"] / a["mcnt"]
        var = (a["sq"] / a["cnt"] - (a["s"] / a["cnt"]) ** 2).clip(lower=0)
        a["ms_std"] = var ** 0.5                            # spread of per-MS load
        a["ms_max"] = a["mx"]                               # busiest single MS
        a["n_active"] = a["nact"]                           # co-bursting MS count
        g = a.rename(columns={"nodeid": "node"})
        g["callrate"] = 0.0
        g["fanout"] = 0.0
        cols = ["node", "t_step", "cpu", "mem", "callrate", "fanout",
                "n_active", "ms_std", "ms_max"]
        return g[cols], dt

    # ---- fallback: node-level Node table (no per-MS features) -----------
    if not node_files:
        raise FileNotFoundError(
            f"No MSResource/MSMetrics or Node CSV under {dd}. Extract the "
            "downloaded .tar.gz into data/ (see DOWNLOAD_REAL_DATA.md).")
    peak = {}
    for p in node_files:
        for ch in _iter(p, NODE_COLS):
            for nid, v in ch.groupby("nodeid")["cpu"].max().items():
                if v > peak.get(nid, -1.0):
                    peak[nid] = v
    top = set(sorted(peak, key=peak.get, reverse=True)[:max_nodes])
    parts = []
    for p in node_files:
        for ch in _iter(p, NODE_COLS):
            ch = ch[ch["nodeid"].isin(top)]
            if len(ch):
                ch = ch.assign(t_step=(ch["timestamp"] / 1000.0 / dt).astype(int))
                parts.append(ch.groupby(["nodeid", "t_step"])
                               .agg(cpu=("cpu", "mean"), mem=("mem", "mean")).reset_index())
    g = (pd.concat(parts, ignore_index=True).groupby(["nodeid", "t_step"])
           .agg(cpu=("cpu", "mean"), mem=("mem", "mean")).reset_index())
    g = g.rename(columns={"nodeid": "node"})
    g["callrate"] = 0.0
    g["fanout"] = 0.0
    return g[["node", "t_step", "cpu", "mem", "callrate", "fanout"]], dt


# ---------------------------------------------------------------- google 2011
def load_google2011(cfg):
    """Loader skeleton for the Google 2011 task_usage tables (CC-BY).

    Expects pre-downloaded task_usage part-*.csv in data_dir. Columns per the
    schema doc: start_time, end_time, machine_id, cpu_rate, canonical_mem_usage.
    Aggregates task usage to node (machine) level at dt.
    """
    import os, glob
    dd = cfg["data"]["data_dir"]
    dt = cfg["data"]["dt_seconds"]
    parts = sorted(glob.glob(os.path.join(dd, "task_usage", "part-*.csv*")))
    if not parts:
        raise FileNotFoundError(
            f"No task_usage parts in {dd}/task_usage. Download via gsutil "
            "(see README Step 1).")
    cols = {0: "start_time", 4: "machine_id", 5: "cpu_rate", 6: "mem"}
    frames = []
    for p in parts[: cfg["data"].get("max_parts", 2)]:
        d = pd.read_csv(p, header=None, usecols=list(cols), names=None)
        d = d.rename(columns={k: v for k, v in cols.items()})
        d["t_step"] = (d["start_time"] / 1e6 / dt).astype(int)   # micros -> steps
        frames.append(d)
    raw = pd.concat(frames, ignore_index=True)
    g = raw.groupby(["machine_id", "t_step"]).agg(cpu=("cpu_rate", "sum"),
                                                  mem=("mem", "sum")).reset_index()
    g = g.rename(columns={"machine_id": "node"})
    # keep the busiest few machines so the demo footprint stays small
    top = g.groupby("node")["cpu"].max().nlargest(cfg["data"].get("n_nodes", 4)).index
    g = g[g["node"].isin(top)].copy()
    g["callrate"] = 0.0
    g["fanout"] = 0.0
    return g[["node", "t_step", "cpu", "mem", "callrate", "fanout"]], dt


def load(cfg, rng):
    mode = cfg["data"]["mode"]
    if mode == "synthetic":
        return make_synthetic(cfg, rng)
    if mode == "alibaba":
        return load_alibaba(cfg)
    if mode == "google2011":
        return load_google2011(cfg)
    raise ValueError(f"unknown data.mode: {mode}")
