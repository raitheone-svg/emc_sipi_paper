# Power-Integrity-Aware Predictive Autoscaling — experiment pipeline

Runnable code for the EMC SIPI 2027 paper. It converts a public cluster trace
into droop/throttle labels (co-simulation), trains a predictor, replays a
droop-aware controller, and emits `results.json` + `fig2_droop.png` that the
paper build reads to **populate Tables I/II and Fig. 2 automatically**.

```
trace ─▶ co-sim (power→PDN→thermal→labels) ─▶ burst features ─▶ predictor
      └▶ controller replay ─▶ outcome ─▶ results.json ─▶ paper
```

## 0. Install
```
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
pip install torch        # OPTIONAL: enables the TCN+attention model (paper model)
```
Without `torch`, the learned model falls back to a gradient-boosted tree
(`backend` is reported in every result, so provenance is never ambiguous).

## 1. Smoke test (no data needed)
```
python run_all.py --config config.yaml      # data.mode: synthetic
```
This generates a labelled synthetic trace and runs the full pipeline. Use it to
confirm the toolchain works and to see the exact output format. **Synthetic
numbers are stamped `SYNTHETIC-DEMONSTRATION` and must never be submitted.**

## 2. Get a real, no-cost dataset
**Alibaba microservice trace** (recommended — free, direct download):
```
git clone https://github.com/alibaba/clusterdata.git
# cluster-trace-microservices-v2022 : fetch MSResource, MSCallGraph, Node
# convert the tables you need to parquet, e.g.:
#   MSResource.parquet  (columns: timestamp, msname, nodeid, cpu, mem)
#   MSCallGraph.parquet (columns: timestamp, um, dm)
# place them in ./data/
```
**Google 2011 Borg trace** (CC-BY — free, attribution only):
```
pip install gsutil
gsutil -m cp -r gs://clusterdata-2011-2/task_usage ./data/task_usage/
```
Edit the loader column names in `src/io_trace.py` to match the exact release you
downloaded (the schema docs in each repo list the fields).

## 3. Run on real data
In `config.yaml` set:
```
data.mode: alibaba          # or google2011
data.dt_seconds: 1.0        # real traces are coarse; 1 s is a good start
controller.provenance: "REAL:Alibaba-microservices-v2022"
```
Sweep the PDN parameters (the one assumption that matters):
```
pdn.sweep.l_henry / r_ohm   # run once per grid point, keep medians
```
Then:
```
python run_all.py --config config.yaml
```
Outputs land in `outputs/`: `results.json`, `tables.md`, `fig2_droop.png`.

## 4. Populate the paper (one command)
From the paper folder (one level up):
```
./build_paper.sh EMC_SIPI_2027_IEEE_template.docx pipeline/outputs/results.json
```
Because `provenance` starts with `REAL:`, the placeholder banner disappears and
the results footnote cites your backend + dataset. Re-render to check, and the
submission `.docx` now carries your measured numbers.

## What each output maps to in the paper
| Pipeline output | Paper element |
|---|---|
| `results.json → table1` | **Table I** (precision / recall / F1 / lead) |
| `results.json → table2` | **Table II** (peak & p99 droop, guardband, throttle, latency, SLO) |
| `outputs/fig2_droop.png` | **Fig. 2** (current + voltage, baseline vs controlled) |
| `backend` field | the model row name + results footnote |

## Honesty checklist before submitting
- [ ] `provenance` says `REAL:<dataset>` (not `SYNTHETIC-DEMONSTRATION`).
- [ ] Results are **medians over the PDN sweep**, not a single lucky point.
- [ ] If you used the GBDT fallback, say so — or install `torch` and report the TCN.
- [ ] Absolute mV figures are framed as model-dependent; headline the **relative** deltas.
- [ ] Ablation (learned vs current-threshold) is reported — that is what earns the ML its place.

## Files
```
config.yaml        all knobs (data, power, PDN, thermal, model, controller)
run_all.py         orchestrator
src/io_trace.py    synthetic + Alibaba + Google 2011 loaders
src/cosim.py       power → PDN → thermal → labels   (the electrical assumptions)
src/features.py    burst-signature windows + forward labels
src/model.py       reactive / forecast baselines + GBDT (+ optional TCN+attention)
src/controller.py  droop-aware replay (stagger + bound) → outcome
src/report.py      Table I / II metrics, Fig. 2, results.json
```
