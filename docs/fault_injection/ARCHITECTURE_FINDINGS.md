# Architecture findings (from source, Stage 1)

All line numbers refer to the local checkout at commit `a4053bc`.

## a. How `current` advances

- `serving/core/controller.py:54-71` `parse_output` matches
  `sys[N] iteration M finished, C cycles, exposed communication X cycles`
  but returns only `{'sys', 'id', 'cycle'}`. The exposed-comm value was
  parsed and then dropped. We extended it to also return `com_cycle`
  in a backward-compatible way (old readers ignore the extra key).
- `serving/__main__.py:686-689` sets `current = out_dict['cycle']` on
  every ASTRA-Sim report. So `current` is the finished iteration's end
  clock in ns.

## b. Where `comp_time` is produced

- `serving/core/trace_generator.py:994` `formatter(...)` builds one row.
  Called from `_emit_layer`, `_emit_moe_block`, `_emit_pim_attention`.
- `serving/core/trace_generator.py:1606-1715` `generate_trace` returns
  a `TraceData` object with `.header_line` and `.rows` (confirmed at
  line 1715). The rows go to `generate_graph`, which hashes them for
  the graph cache. So mutating rows after `generate_trace` and before
  `generate_graph` changes the cache key naturally.

## c. P/D KV transfer path (RESOLVED)

- `serving/core/request.py:119-136` `Batch.pd_kv_send_tokens` counts
  tokens whose KV must be shipped. Set in `scheduler.py:293-314`.
- `serving/core/trace_generator.py:1047-1059` puts the KV byte count
  on the `comm_size` column of `qkv_proj` rows, only for prefill.
- `serving/core/config_builder.py:592-594` a prefill instance owns
  `num_npus * 2` NPUs (`effective_npus`). The extra range is its
  own receiver.
- `astra-sim/extern/graph_frontend/chakra/.../llm_converter.py:755`
  the sender file is `npu_id + num_npus`, i.e. inside the prefill
  instance's own range, NOT the real decode instance.

CONFIRMED: the modeled transfer link is per prefill instance and is
decoupled from which decode instance the router later picks. Per-pair
(prefill i to decode j) transfer targeting is NOT possible. The schema
therefore targets the prefill instance's transfer path, documented as
a limitation.

## d. Request/Batch timestamps

- `serving/core/request.py:39-43` has `queuing_delay`, `ttft`, `tpot`,
  `itl`, `recent_end`. `batch_time` lives on `Batch` (line 124).
- `set_que_delay`, `set_ttft`, `add_itl`, `add_latency` are plain
  setters. `set_ttft` also stamps `recent_end`, so the first decode
  ITL includes the gap since TTFT.
- In PD mode a prefill batch completes only when both the start NPU
  and the receiver NPU report, so TTFT includes the transfer time.
- `kv_handoff_delay` delays decode admission, so it moves the first
  ITL but not TTFT. This is how we separate the two transfer faults.

## e. network.yml / system.json

Generated fresh on every run by `config_builder.py`. ASTRA-Sim reads
them once at launch. Nothing changes link bandwidth or latency
mid-run. So time-varying link latency cannot be mechanism-level
without changing ASTRA-Sim. We label that `output_approx` only.

## f. Pitfalls that touch this work

- DP barrier: every DP member must run every wave, dummy batches
  included. Prototype rejects `dp_group` with faults (clear error)
  instead of risking the silent deadlock in AGENTS.md.
- `pass` / `pass <tick>` suppression: fault start/end and handoff
  release times are wake-up events; they are folded into the
  clock-advance logic, otherwise an all-idle cluster sleeps past them.
- Exit conditions: `has_held_transfers` is part of every done check.
- `add_done` needs start and last NPU; stalled instances simply
  schedule nothing, in-flight batches still complete.

## Capability table

| Capability | Status | Where / why |
|---|---|---|
| Additive compute slow | NATIVE via extension | `fault_injection.apply_to_trace_rows`, `__main__` hook |
| Multiplicative compute slow | NATIVE via extension | same as above |
| Time-varying profile | NATIVE via extension | `severity_weight` in same file |
| Transfer bandwidth | NATIVE via extension | `comm_size` inflation on `qkv_proj` |
| Handoff delay | NATIVE via extension | Router held queue in `__main__` |
| Worker stall | NATIVE via extension | router skip + schedule guard, non-DP only |
| Targeting by role+instance | NATIVE via extension | `fault_matches`; per-pair transfer NOT FEASIBLE (see c) |
| Per-stage telemetry | NATIVE via extension | `--telemetry-output` CSVs, original CSV untouched |
| Time-varying link latency | OUTPUT-LEVEL ONLY | would need ASTRA-Sim change; labeled `output_approx` |
| Real crash with loss | NOT FEASIBLE | out of scope; in-flight batches complete |
