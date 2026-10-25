# Smoke experiment (reproducibility bundle)

Preliminary small-scale end-to-end check of the fault-detection
pipeline. This is NOT the final benchmark — it exercises the full
path (workload → simulate → manifest → features → labels → baselines →
report) on 5 short runs so the machinery, not the science, is what gets
tested.

> **Two things in this directory have different standing.**
>
> 1. The **five clocks** are the live regression baseline and are still
>    exact after the consolidation and the instrument repairs
>    (7667807319 / 8387062877 / 4862442514 / 8794093936 / 8894547812 ns,
>    re-measured twice: sequential and with 3 parallel workers).
> 2. The frozen **`metrics.csv` is historical**. It was produced by the
>    pre-repair instrument, which labelled every window inside the
>    injection schedule anomalous and whose localiser echoed the ground
>    truth. Re-running the current `evaluate.py` on this same frozen
>    `data/` yields 33 windows (8 `root_cause`, 25 `unaffected`, 7
>    `downstream` excluded) instead of "40 windows, 22 anomalous". Do not
>    cite the old rows; the current numbers come from `review_min`
>    (`report.md` §16.5).

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
- 5 runs, 0 failed.

## Exact command that produced this bundle

From the repo root, inside the sim container:

```bash
python -m experiments.fault_detection.inject_faults \
  --matrix experiments/fault_detection/matrix.yaml \
  --profile smoke --out experiments/out/
```

Seed: 42 (`--seed 42` default). The live manifest is
`experiments/out/manifest.jsonl` (gitignored). This directory holds a
frozen copy laid out exactly as the runner expects: `data/` contains
`manifest.jsonl` plus one directory per `pair_id` (shared workload,
per-twin telemetry, audit); `metrics.csv` is the report output frozen
alongside for row-for-row comparison (historical, see the note above).

## Regenerate metrics from the frozen data (no simulator needed)

```bash
python -m experiments.fault_detection.evaluate \
  --data experiments/fault_detection/smoke/data --report <empty-dir>/
```

Requires only Python with numpy. Output: `metrics.csv`, `windows.csv`,
`report.md`. `inference_ms_per_window` is wall-clock timing and varies by
machine; every other row is deterministic.

## Expected outputs and layout

- `data/<pair_id>/workload.jsonl` — shared arrivals for the twins
- `data/<pair_id>/{healthy,faulty}_telemetry/` — observable
  `batch_telemetry.csv`, `request_telemetry.csv`, `request_itls.jsonl`
  plus the ground-truth `injection_audit.jsonl` (never fed to features)
- `manifest.jsonl` — one line per run (clocks, fault detail,
  `successfully_applied`, expected label, status, regime)
- `metrics.csv` (historical) — detector metrics; `report.md` — verified
  numbers with raw counts; `architecture.svg` — system diagram

## Why this bundle is small

40 windows cannot separate "residuals help" from "this dataset is easy".
Its job is to prove the pipeline runs end to end and that the simulator
still produces the exact clocks it produced before. The scientific
measurement is the calibrated `review_min` matrix.

## Hardware counters

The detector uses simulator-visible serving telemetry only (batch
durations, queues, stage stamps, TTFT/ITL stats, KV-send counts,
exposed-communication proxy). Physical hardware counters from the
broader problem statement (GPU power/thermal, SM/GPU utilisation,
NIC/RDMA counters, clocks/throttling) are not exposed through the
simulator detection path.
