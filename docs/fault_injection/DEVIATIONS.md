# Deviations from the task prompt (local code wins)

The prompt says: where the prompt and the local code disagree, the local
code wins and the discrepancy is recorded here and in DESIGN.md.

1. Additive compute delay is per ITERATION (one serial row), not per layer.
   An early prototype added delta_ns to every row, which would multiply
   the delay by the layer count. Fixed in `apply_to_trace_rows`; unit test
   `test_additive_applies_once_per_iteration_not_per_layer` pins it.
2. `kv_transfer_bandwidth` inflation is weight-aware
   (`1 + w(t)*(1/fraction - 1)`), so partial weights (ramps,
   intermittent off-phases) interpolate instead of jumping. The prompt's
   `1/bandwidth_fraction` is the special case w(t) = 1.
3. `network_degradation` (mechanism) scales `comm_size` on collective
   rows, never `comp_time`. An early prototype scaled comp_time as a
   placeholder; that conflated network faults with compute faults in the
   audit and was replaced.
4. `transfer_half_bw` on `example_trace.jsonl` shows +0.0% clocks (tiny
   KV bytes are invisible at ns resolution). Reported as an honest
   negative result; transfer calibration needs a long-prompt workload.
5. Per-pair P/D transfer targeting is NOT possible (Stage 1c confirmed:
   the receiver NPU lives in the prefill instance's own range). The schema
   targets the prefill instance's transfer path; documented as a limitation.
6. Whole-run static link override via `config_builder._create_network_config`
   is not wired to fault entries. The cluster config's own per-dimension
   `link_bw`/`link_latency` arrays cover the static case; fault entries
   cover the time-varying case. Wiring the two together is deferred.
7. `compute_additive` on a single-row unit-test trace adds delta to that
   row (it is both first and last); multi-row behavior is pinned by the
   dedicated test in (1).
8. Validate.sh fault-scenario baselines are recorded with faults active
   and therefore intentionally differ from the plain `pd` clock. The
   transfer-bw scenario may equal `pd` on short-prompt traces (see 4);
   that equality is itself a recorded finding, not coverage padding.
