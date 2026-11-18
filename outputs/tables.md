Provenance: REAL:Alibaba-microservices-v2022  |  backend: TCN+attention (PyTorch)

Table I — event prediction
model | precision | recall | f1 | lead(ms)
Reactive di/dt threshold | 0.25 | 0.69 | 0.37 | 120000
Forecast + threshold | 0.47 | 0.65 | 0.54 | 120000
TCN+attention (PyTorch) | 0.3 | 0.42 | 0.35 | 120000

Table II — power-integrity outcome
Peak droop (mV) | 33 | 33 | +0%
p99 droop (mV) | 24 | 25 | +1%
Reclaim. guardband (mV) | — | 0 | +0
Throttle events / hr | 0.0 | 0.0 | —
p99 latency cost (ms) | — | +100497.6 | —
SLO violation rate | 1.00× | 0.92× | -8%
