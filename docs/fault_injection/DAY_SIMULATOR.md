# Day simulator: multi-GPU P/D serving with synthetic throttling (build notes)

Student-readable companion to the plan
(`docs/RESOURCE_AWARE_SIMULATOR_PLAN.md`). Implementation authorised in
Build mode; container sim runs pending (Docker daemon down at build time).

## What was built

- `experiments/fault_detection/day_workload.py`: NHPP arrival generator
  (morning ramp, configurable Gaussian noon peak, afternoon decay, evening
  level, optional bursts), seeded, existing JSONL schema. 24h generates in
  ~0.6s wall on host; SIMULATING a day costs ASTRA-Sim handshakes per
  batch -- measure with short horizons first.
- `serving/core/gpu_state.py`: per-GPU synthetic accumulated-load state
  (`GpuThrottleState`, `ThrottleManager`, `apply_throttle_factor`,
  config validation/loading). Disabled-by-default; `factor()` is 1.0
  unless throttled.
- `serving/__main__.py` wiring: `--throttle-config`,
  `--throttle-audit-output`, `_throttle_tick`, build-start record, factor
  hook after the fault hook, completion-window busy accounting,
  `throttle_audit.jsonl` write-out.
- `configs/cluster/moe_pd_tp2_pp2_instance.json`: 1P1D P/D topology on
  Qwen3-30B-A3B, TP=2 PP=2 EP=2 per pool (8 instance-GPUs, 12 NPUs with
  prefill shadow range).
- `experiments/fault_detection/example_throttle/decode_pool_moe_pd_tp2_pp2.json`:
  decode-pool-only example (GPUs 8-11).
- Tests: `test_day_workload.py`, `test_gpu_throttle.py`,
  `test_router_paths.py` (absolute-path regression). 36 passed on host.
- Bug fix: `Router.load_requests` now honours absolute dataset paths
  (`os.path.isabs` guard); relative `"../"` behaviour unchanged.

## Architecture and assumptions

- GPU = physical NPU index. `instance_to_gpus` maps each instance to
  `[start, start+num_npus)`; a prefill instance's shadow receiver range
  is transfer fiction, excluded. Uniform `window_s` enforced across GPUs.
- Busy fraction `u` = completed-batch durations in window / window,
  capped at 1.0, attributed to the completion window. Never queue length
  (circular). Under lockstep SPMD all member NPUs share the duration.
- Throttle factor = max over member GPUs, applied to FUTURE batches at
  schedule time (in-flight ASTRA-Sim work cannot be rescaled -- same
  limitation as fault injection). Compute rows only; `comm_size` untouched.
- End-of-day: the sim drains naturally (done conditions require empty
  queues); arrivals stop at the horizon. No silent drops by construction;
  size grace by workload horizon, not a separate flag (unimplemented).
- Ground truth (`throttle_audit.jsonl`: per-window load/busy/throttled/
  factor + per-batch scaled rows) is evaluation-only, never a detector
  feature. `ml/features.py` untouched.

## Model profile and parallelism (verified, not assumed)

- Qwen3-30B-A3B-Instruct-2507 (`qwen3_moe`, 128 experts, top-8, 48
  layers) on RTX PRO 6000. Profiles: tp1+tp2 dense/attention/
  per-sequence/skew; `moe.csv` tp1-only -- MoE timing at TP=2 is
  analytic per-EP-rank lookup, NEVER claimed as independently measured.
- EP=2=TP satisfies `ep<=tp` (no dp_group), 128%2==0, pp=2<=48.
- PP=4 MoE: UNVERIFIED (no config, no validation run -- not attempted).

## Thermal-proxy limitations (synthetic, not physical)

`L` is a dimensionless workload-history accumulator, NOT temperature. No
power, heat dissipation, ambient, or temperature model exists here; no
real-HW calibration. Do not describe throttling as thermal behaviour of
a real RTX PRO 6000. Synthetic data cannot validate real-hardware
detection.

## Known gaps (container runs pending)

- Deterministic M1 smoke on `moe_pd_tp2_pp2` (needs ASTRA-Sim binary).
- M6 sim comparisons (healthy/peak/throttled, P-vs-D, TP=2/PP=2 vs
  alternative parallelism) with normal-vs-slow tables.
- `serving/validate.sh` regression + fault-detection smoke rerun.
- DP + throttle explicitly excluded (existing DP+faults guard carries over).
- `ThrottleManager.window_s` takes the last GPU's value; multi-window
  configs are rejected at startup instead.
