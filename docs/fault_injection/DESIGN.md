# Design note (prototype, student-readable)

## What each mode does and where it hooks

1. `compute_multiplicative` / `compute_additive`
   Hook: `serving/__main__.py` after `generate_trace`, before
   `generate_graph`, via `fault_injection.apply_to_trace_rows`.
   Why: the trace rows are the only place profiled latency enters.
   Scaling `comp_time` there changes what ASTRA-Sim executes.
   Integer-rounded because the converter requires int.
   MoE/PIM rows are scaled too when present; nothing special.

2. `kv_transfer_bandwidth`
   Hook: same function, `comm_size` on `qkv_proj` rows inflated by
   `1 / bandwidth_fraction`. This slows the real SEND/RECV, so the
   prefill iteration, TTFT and sender occupancy grow.
   Models only the bandwidth term, not latency. Microbenchmark
   (fixed workload, scale 2x/4x) should be recorded here once the
   sim container is available; on this host the binary is missing
   so no numbers are claimed.

3. `kv_handoff_delay`
   Hook: Router-side held queue in `__main__` with release times.
   Delays `add_decode`, so first ITL moves but TTFT does not.
   Needs: held queue, done-condition inclusion, pass-deadline
   inclusion. All three are implemented; missing any one hangs
   or exits early (regression test to be added with binary).

4. `network_degradation`
   Static per-dimension `link_bw` arrays are already supported by
   the cluster config. Time-varying is done as `comm_size` scaling
   (same hook as 2). Time-varying link *latency* is output-level
   only (`injection_layer: output_approx`), excluded from
   mechanism-level evaluation.

5. `worker_stall`
   Hook: router skips stalled instances, `schedule` guarded in
   `__main__`. In-flight batches complete; queued requests wait.
   Non-DP only; `dp_group` + faults raises a clear error.
   Near-stop fail-slow (factor 10+) is provided for
   crash-vs-fail-slow comparison. Real crash with loss is out
   of scope.

## Timing semantics

Fault time is evaluated at the `current` clock passed into
scheduling for that batch. Batches already submitted are never
altered. Onset error is at most one iteration duration.

## What it cannot model

Batch-granularity onset; no mid-iteration faults; transfer time
not separable natively (proxy: exposed-comm cycles + TTFT split);
link latency not time-varying; per-prefill-instance transfer path;
synthetic faults are not real hardware failures.

## Discrepancies (local code wins)

- Prompt said `controller.parse_output` returns sys/id/cycle and
  matches exposed-comm but drops it. True. We added `com_cycle`
  as an extra key (backward compatible).
- Prompt assumed per-pair transfer targeting. Local code refutes
  it (see ARCHITECTURE_FINDINGS c). Schema targets the prefill
  instance instead.
