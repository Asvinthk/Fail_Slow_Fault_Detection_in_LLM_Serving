# Fail-slow fault injection (prototype)

Student-readable extension to LLMServingSim 2.0 for PD-disaggregated
fail-slow research. The custom simulation logic lives in one file:
`serving/core/fault_study.py` (injection, synthetic throttling, benign
jitter, audit). The experiment harness is three plain-Python modules under
`experiments/fault_detection/`.

## Install / run

The simulator runs in the sim container:

```bash
scripts/docker-sim.sh
scripts/compile.sh
python -m serving --cluster-config configs/cluster/pd_1p1d_small.json \
  --dtype bfloat16 --block-size 16 --dataset workloads/example_trace.jsonl \
  --output outputs/fi_healthy.csv --num-reqs 10 --run-id fi_healthy
```

With a fault (mechanism = simulator injection mode only):

```bash
python -m serving --cluster-config configs/cluster/pd_1p1d_small.json \
  --dtype bfloat16 --block-size 16 --dataset workloads/example_trace.jsonl \
  --output outputs/fi_faulty.csv --num-reqs 10 --run-id fi_faulty \
  --fault-config experiments/fault_detection/example_faults/decode_slow_2x.json \
  --telemetry-output experiments/out/fi_faulty_telemetry
```

Without `--fault-config`, `--throttle-config` or `--timing-jitter`, no new
code path executes; `validate.sh` baselines are byte-identical.

## Pipeline (three commands)

```bash
# load-matched pilot: find lambda for a target decode utilisation
python -m experiments.fault_detection.run_simulation calibrate \
  --cluster configs/cluster/pd_2p2d_small.json --target-u 0.75 \
  --out experiments/out/pilot_2p2d_u75

# paired healthy/faulty matrix (same trace, one healthy twin per group)
python -m experiments.fault_detection.inject_faults \
  --matrix experiments/fault_detection/matrix.yaml \
  --profile review_min --out experiments/out/review_min --seeds 1,2,3,4,5

# features + audit labels + transparent baselines + CIs + report
python -m experiments.fault_detection.evaluate \
  --data experiments/out/review_min --report experiments/report/review_min
```

`notebooks/train_detector.ipynb` is the only notebook and only trains learned
detectors on an already-validated `windows.csv`.

## Outputs

- Original CSV (`--output`): unchanged schema.
- `batch_telemetry.csv` / `request_telemetry.csv` under `--telemetry-output`:
  observable serving telemetry.
- `injection_audit.jsonl` and `throttle_audit.jsonl` next to telemetry:
  ground truth in separate files, never merged into telemetry.
- `manifest.jsonl` per matrix run: fault config, seed, regime
  (`lambda`, `U_prefill`, `U_decode`, TPOT), noise level, git commit.

## Limitations

Synthetic faults do not prove real-hardware detection. "Mechanism" =
simulator injection mode only. See DESIGN.md for the full list (batch
granularity, transfer separability, per-prefill-instance path, no mid-run
link change) and `report.md` §16 for the current instrument's known gaps.
