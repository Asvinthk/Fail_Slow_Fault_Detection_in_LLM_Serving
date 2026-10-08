# Fault config schema

JSON file passed as `--fault-config <path>`. Times in seconds,
converted to ns internally (`current / 1e9`).

```json
{
  "faults": [
    {
      "fault_id": "decode0_slow_2x",
      "target": {"role": "decode", "instance_id": 1},
      "injection_layer": "mechanism",
      "mode": "compute_multiplicative",
      "factor": 2.0,
      "schedule": {"start_s": 1.0, "end_s": 30.0, "profile": "step"}
    }
  ]
}
```

- `target.role`: prefill | decode | transfer | network.
  `transfer` means the prefill instance's transfer path (per-pair
  targeting is not possible, see ARCHITECTURE_FINDINGS).
  `network` with `instance_id: "all"` hits every instance.
- `target.instance_id`: int (cluster config order) or `"all"`.
- `mode` parameters:
  - `compute_multiplicative` needs `factor` (e.g. 1.1, 1.25, 1.5, 2.0).
  - `compute_additive` needs `delta_ns`.
  - `kv_transfer_bandwidth` needs `bandwidth_fraction` (0.5 = half).
  - `kv_handoff_delay` needs `delay_ns`.
  - `worker_stall` takes no size parameter.
  - `network_degradation` uses `factor`; link-latency variant must
    set `injection_layer: "output_approx"` and is excluded from
    mechanism-level claims.
- `schedule`: `start_s`, `end_s`, `profile`:
  `step` | `intermittent` (`period_s`, `duty`) |
  `ramp_up` / `ramp_down` / `ramp_up_down` (`ramp_s`).
  Severity `w(t)` in [0,1]; effective factor `1 + w*(factor-1)`,
  effective additive `w*delta`.
- Unknown keys or bad combos are rejected at startup with `ValueError`.
- `worker_stall` with `dp_group` configs is rejected (clear error).

Audit: `injection_audit.jsonl` next to `--telemetry-output` when given,
else next to `--output` (never under the auto-deleted inputs root).
One line per affected batch plus handoff holds. `successfully_applied`
(manifest) is true only if the audit has a record after the fault start.
