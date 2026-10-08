# Fail-slow fault injection (prototype)

Student-readable extension to LLMServingSim 2.0 for PD-disaggregated
fail-slow research. One new source file: `serving/core/fault_injection.py`.

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

Without `--fault-config` no new code path executes; `validate.sh`
baselines are byte-identical.

## Outputs

- Original CSV (`--output`): unchanged schema.
- `experiments/out/.../batch_telemetry.csv`: observable per-batch stats.
- `injection_audit.jsonl` next to `--telemetry-output` (or `--output`
  without it): ground truth, separate file, never in telemetry.

## Limitations

Synthetic faults do not prove real-hardware detection.
"Mechanism" = simulator injection mode only.
See DESIGN.md for the full list (batch granularity, transfer
separability, per-prefill-instance path, no mid-run link change).
