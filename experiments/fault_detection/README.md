# Fault-detection experiments

Research extension on top of LLMServingSim 2.0. The project **extends** the
simulator (fault hooks, telemetry, experiment harness) rather than replacing
its scheduler or network backend, which are upstream code and stay
byte-identical when no fault option is given.

## Four files, four jobs

| File | Job |
|---|---|
| `run_simulation.py` | Generate workloads (Poisson / bursty / sustained / NHPP day), measure the regime of a finished run, calibrate `lambda` for a target decode utilisation, read manifests and audits, invoke `python -m serving` once. |
| `inject_faults.py` | Expand `matrix.yaml` into paired healthy/faulty runs, write fault and throttle configs, keep ground truth in `manifest.jsonl`. |
| `evaluate.py` | Clean telemetry, build per-(window, instance) features, derive labels from the audit, run transparent baselines (robust threshold / residual / EWMA / CUSUM), localise with abstention, write `metrics.csv`, `windows.csv`, `report.md`. |
| `matrix.yaml` | All workload, cluster, severity, bandwidth, noise and seed settings. |

The only notebook is `notebooks/train_detector.ipynb`, and it only trains and
compares learned detectors on an already-validated `windows.csv`.

## Commands

```bash
# 1. calibrate load: pick lambda for a target decode utilisation
python -m experiments.fault_detection.run_simulation calibrate \
    --cluster configs/cluster/pd_2p2d_small.json --target-u 0.75 \
    --out experiments/out/pilot_2p2d_u75

# 2. paired matrix (healthy + faulty twins on one trace), 5 seeds
python -m experiments.fault_detection.inject_faults \
    --matrix experiments/fault_detection/matrix.yaml \
    --profile review_min --out experiments/out/review_min --seeds 1,2,3,4,5

# 3. features, labels, baselines, CIs, report
python -m experiments.fault_detection.evaluate \
    --data experiments/out/review_min --report experiments/report/review_min
```

Every run is resumable: `inject_faults` skips pairs whose runs are already
`ok` in the manifest, and re-running with the same seeds reproduces the same
traces.

## What a result means

- Numbers come from **synthetic faults in a simulator**. They validate the
  detection machinery; they are not evidence of real-hardware detection.
- Every reported number carries its regime (`lambda`, `U_prefill`,
  `U_decode`, TPOT), its run count, and a confidence interval over runs
  (Wilson for proportions, bootstrap over runs for P/R/F1/FPR).
- Labels are derived from the injection audit, per (window, instance), and
  the same clock (`finish_ns`) buckets features and labels.
- Downstream / recovering / neutral windows are reported separately and
  excluded from the detection metric, never counted as false alarms.
- A fault family whose severities produce bit-identical total clocks is
  **timing-neutral**: the trace changed, the simulation did not. Those
  windows are excluded as `neutral` and the family is named in
  `report.md`, because scoring them would count a guaranteed miss against
  every detector. `kv_transfer_bandwidth` is such a family at tp=1.
- A run whose telemetry directory cannot be found is reported
  (`n_runs_missing_telemetry`), never silently dropped and never treated
  as healthy.

## Layout notes

- `smoke/` — frozen reproducibility bundle for the smoke evaluation
  (5 runs, clocks recorded in its manifest).
- `example_faults/`, `example_throttle/` — small committed configs for
  manual runs.
- Generated data stays gitignored (`experiments/out/`, `experiments/report/`).
- The manifest records each run's `telemetry_dir`, so a pair with several
  fault variants (`faulty0_telemetry`, `faulty1_telemetry`, ...) is read
  correctly instead of being derived from `fault_present`.

## Honest scope notes

- Localisation is per-instance residual ranking with abstention, not a
  trained classifier; top-1/top-3 are reported with a Wilson interval.
  The candidate set is NPU instances, so a transfer-link label
  (`kv-transfer-network:0`) is reported as `localization_not_localisable`
  rather than scored -- its id shares no space with NPU ids.
- `mean_detection_delay_s` is an upper bound, not a detection latency: the
  generated report carries a per-family table showing which families
  alarmed while their audit says nothing was applied or their clocks were
  bit-identical, and says so explicitly next to the number.
- The detector uses simulator-visible serving telemetry only. Physical
  hardware counters (GPU power/thermal, SM utilisation, NIC/RDMA) are not
  exposed through the simulator path.
- "Mechanism" means simulator injection mode, never physical root cause.
- At `--timing-jitter 0` the simulator is deterministic, so healthy
  residuals have almost no spread; detectability is over-optimistic unless
  a benign noise level is set. Sweep `--timing-jitter` (0.02/0.05/0.10).
