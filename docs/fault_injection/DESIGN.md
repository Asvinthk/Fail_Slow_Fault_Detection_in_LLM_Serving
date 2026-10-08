# Design note (prototype, student-readable)

## What each mode does and where it hooks

1. `compute_multiplicative` / `compute_additive`
   Hook: `serving/__main__.py` after `generate_trace`, before
   `generate_graph`, via `fault_injection.apply_to_trace_rows`.
   Why: the trace rows are the only place profiled latency enters.
   Scaling `comp_time` there changes what ASTRA-Sim executes.
   Integer-rounded because the converter requires int.
   Multiplicative scales every compute row by the effective factor
   `1 + w(t)*(factor-1)`. Additive is per ITERATION: the weighted
   delta is added to ONE serial row (the last 11-field row), never
   per layer. MoE EXPERT / PIM marker rows (not 11-field) are left
   untouched. Measured in-container (single_node_pd_instance,
   example_trace, 10 reqs): healthy 1661784496 ns (x2 identical);
   decode-1 factor 2.0 -> 2318421829 ns (+39.5%, x2 identical),
   audit 59 records all on instance 1 with per-batch comp sums
   exactly doubled (total 656787414 -> 1313574828). A pre-fix record
   of 2318732875 ns for the same fault differs by 0.013%: the
   fault-boundary wake-ups added since shift idle polling slightly;
   the per-batch injection is unchanged (ratio exactly 2.0).

2. `kv_transfer_bandwidth`
   Hook: same function, `comm_size` on `qkv_proj` rows inflated by
   the effective factor `1 + w(t)*(1/fraction - 1)`. This slows the
   real SEND/RECV, so the prefill iteration, TTFT and sender
   occupancy grow. Models only the bandwidth term, not latency.
   Calibration (1 request, 2048-token prompt, 256 MB KV):
   - link 16 GB/s: healthy 255189132 ns; 2x and 4x both identical.
     The receiver-side transfer overlaps sender compute (sender
     ~200 ms vs transfer ~16 ms), so inflation is fully hidden.
   - link 1 GB/s (`pd_1p1d_slowlink.json`): healthy 423217485 ns
     (+66% vs normal link); 2x -> 673217485 ns (+250 ms, +59%);
     4x -> 1173217485 ns (+750 ms, +177%). Linear in
     (1/fraction - 1) as the bandwidth term predicts.
   Rule: the fault bites only when transfer time is comparable to
   prefill compute (slow link or short compute). On a fast link with
   compute-dominated prefill it is correctly neutral, and the audit
   (comm sums exactly doubled) proves the bytes were inflated.

3. `kv_handoff_delay`
   Hook: Router-side held queue in `__main__` with release times.
   Delays `add_decode`, so first ITL moves but TTFT does not.
   Needs: held queue, done-condition inclusion (`has_held`), release
   time in the `pass <tick>` deadline AND in the idle clock-advance,
   plus fault start/end times as wake-up events (an all-idle cluster
   must wake at stall end, or recovery is never observed). All are
   implemented; missing any one hangs or exits early. Unit-tested
   via `_pass_response` deadline tests; the all-idle binary
   regression run is pending container time.

4. `network_degradation`
   Static per-dimension `link_bw` arrays are already supported by
   the cluster config. Time-varying scales `comm_size` on every row
   carrying a collective (`comm_type != NONE`, `comm_size != 0`) for
   the targeted instance (network targets match by instance
   regardless of role). Time-varying link *latency* is output-level
   only (`injection_layer: output_approx`), excluded from
   mechanism-level evaluation. Whole-run static link overrides via
   `config_builder._create_network_config` are not wired to faults;
   use the cluster config's own per-dimension arrays instead.

5. `worker_stall`
   Hook: router skips stalled instances, `schedule` guarded in
   `__main__`. In-flight batches complete; queued requests wait.
   Transfers to a stalled decode are HELD in the router and retried
   after recovery, never dropped (an early version silently lost one
   request per all-stalled window: 9/10 completed; fixed and
   unit-tested). Non-DP only; `dp_group` + faults raises a clear
   error. Measured (10 reqs, decode-1 stalled 1-5 s): makespan
   5657635463 ns (+3.996 s, the window), audit 1 stall-window record
   (3987 skipped polls), telemetry 0 batches on instance 1 in-window
   and 10/10 requests completed. Near-stop fail-slow (factor 10+) is
   provided for crash-vs-fail-slow comparison. Real crash with loss
   is out of scope.

## Timing semantics

Fault time is evaluated at the `current` clock passed into
scheduling for that batch. Batches already submitted are never
altered. Onset error is at most one iteration duration.
Fault-active runs carry a small observer effect on untargeted
requests (single-digit ms batching shifts from clock-advance wake-ups
and pass deadlines; measured mean TTFT +4.38 ms under a handoff fault
whose mechanism moves latency +100 ms). The disabled path (no
`--fault-config`) is exactly identical in clocks and CSV.

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
