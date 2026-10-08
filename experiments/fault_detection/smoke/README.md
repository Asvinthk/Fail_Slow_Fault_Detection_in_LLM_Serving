# Smoke experiment (reproducibility bundle)

Preliminary small-scale end-to-end check of the fault-detection
pipeline. This is NOT the final benchmark — it exercises the full
path (generate → simulate → manifest → features → baselines → report)
on 5 short runs so the machinery, not the science, is what gets tested.

## What it tests

Paired healthy/faulty runs on a 1P1D cluster: a decode ×2.0 slowdown
on a light workload, and a 100 ms KV-handoff delay on a bursty
workload, each with a healthy twin sharing arrivals and seed; plus a
healthy heavy-load run that specifically probes workload-vs-fault
confusion (heavy-but-healthy must not alarm).

## Workload families and runs

- `light_poisson` (10 reqs, 1/s): healthy + faulty (decode_mult_2x)
- `heavy_poisson` (24 reqs, 8/s): healthy only
- `bursty` (24 reqs, on/off): healthy + faulty (handoff_100ms)
- 5 runs, 0 failed; 40 windows (22 healthy, 18 anomalous) at 2 s windows.

## Exact command that produced this bundle

From the repo root, inside the sim container:

```bash
python -m experiments.fault_detection.generate --matrix experiments/fault_detection/matrix.yaml --profile smoke --out experiments/out/
```

Seed: 42 (`--seed 42` default). The live manifest is
`experiments/out/manifest.jsonl` (gitignored). This directory holds a
frozen copy laid out exactly as the runner expects: `data/` contains
`manifest.jsonl` plus one directory per `pair_id` (shared workload,
per-twin telemetry, audit); `metrics.csv` is the generated report
output frozen alongside for row-for-row comparison.

## Regenerate the ML metrics from the frozen data (no simulator needed)

```bash
python -m experiments.fault_detection.ml.run --data experiments/fault_detection/smoke/data --report <empty-dir>/
```

Requires only Python with numpy (scikit-learn optional; the logistic
baseline falls back to residual predictions without it). The output
`metrics.csv` must match the committed `metrics.csv` on every row
except `inference_ms_per_window`, which is wall-clock timing and
varies by machine (verified: all scientific rows identical;
1.318 ms here vs 4.049 ms in the container).

## Expected outputs and layout

- `data/<pair_id>/workload.jsonl` — shared arrivals for the twins
- `data/<pair_id>/{healthy,faulty}_telemetry/` — observable
  `batch_telemetry.csv`, `request_telemetry.csv`, `request_itls.jsonl`
  plus the ground-truth `injection_audit.jsonl` (never fed to features)
- `manifest.jsonl` — one line per run (clocks, fault detail,
  `successfully_applied`, expected label, status)
- `metrics.csv` — detector metrics; `report.md` — verified numbers
  with raw counts (see below); `architecture.svg` — system diagram

## Verified result (from `report.md`, counts included)

- Fixed threshold: test P 1.000 / R 0.556 / FPR 0.000 (0 false alarms;
  misses the subtle handoff fault).
- Residual-only: test P 0.857 / R 0.667 / F1 0.750 / FPR 0.125, i.e.
  1 false alarm in 8 healthy test windows.
- Logistic regression: test P 0.286 / R 0.222 — preliminary negative;
  insufficient data is a possible explanation, not a proven diagnosis.
- The sample (40 windows) is consistent with workload-normalised
  residuals helping separate faults from workload variation, but too
  small to establish superiority. No general "residuals are better"
  claim is made.
- Localisation in this bundle is a STUB that echoes ground truth when
  confident (`ml/run.py`); its 0.475 figure is not a model result.
  No MLP, gradient boosting, macro-F1, confusion-matrix, top-1/top-3,
  per-family-FPR, or mechanism-classification model is implemented.

## Hardware counters

The detector uses simulator-visible serving telemetry only (batch
durations, queues, stage stamps, TTFT/ITL stats, KV-send counts,
exposed-communication proxy). Physical hardware counters from the
broader problem statement (GPU power/thermal, SM/GPU utilisation,
NIC/RDMA counters, clocks/throttling) are not exposed through the
simulator detection path.
