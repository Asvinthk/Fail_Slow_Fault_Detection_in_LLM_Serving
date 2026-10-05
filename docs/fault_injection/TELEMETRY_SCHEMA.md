# Telemetry schema (observable only)

Enabled by `--telemetry-output <dir>`. The original `--output` CSV
is byte-identical with telemetry on or off.

## batch_telemetry.csv (OBSERVABLE)

One row per ASTRA-Sim iteration report (one per NPU report, so a batch
with TP>1 or a prefill sender+receiver yields several rows sharing a
batch_id). Shape columns come from a schedule-time snapshot
(`batch_meta` in `__main__.py`); `add_done` only sees
`(id, sys, finish)`, so the snapshot is remembered per
`(instance_id, batch_id)`.

| column | measured / estimated |
|---|---|
| run_id | measured (CLI) |
| instance_id | measured (NPU mapping) |
| batch_id | measured (ASTRA-Sim report id) |
| sys | measured (reporting NPU) |
| batch_time_ns | measured (schedule-time clock) |
| finish_ns | measured (report clock) |
| duration_ns | measured (finish - batch_time) |
| num_prefill / num_decode / total_len | measured (batch snapshot at schedule time) |
| kv_len | measured (batch snapshot) |
| pd_kv_send_tokens | measured (batch snapshot) |
| n_waiting / n_running | measured (scheduler queue lengths at schedule time) |
| com_cycle | measured (exposed communication cycles from `controller.parse_output`; simulator-instrumented proxy for transfer-exposed time, NOT a direct transfer-duration measurement) |

KV-transfer time is NOT directly separable from prefill iteration
time in the native simulator. Proxy used: exposed-comm cycles plus
the TTFT-vs-first-ITL split (handoff faults move ITL only).

## request_telemetry.csv (OBSERVABLE)

One row per finished request (from every scheduler's `done` list).
New fields live in `Request.__init__` (no `getattr` fallbacks);
stamps are written only when `--telemetry-output` is given, so the
disabled path is untouched.

| column | measured / estimated |
|---|---|
| request_id | measured |
| prefill_instance_id / decode_instance_id | measured (which pool finished each stage; -1 if never) |
| input_toks / output_toks | measured |
| arrival_ns | measured |
| first_scheduled_ns | measured (first batch the request was built into; queueing end) |
| prefill_done_ns | measured (prefill completion = TTFT stamp) |
| decode_admit_ns | measured (transfer into decode, or handoff release time) |
| first_decode_done_ns | measured (first decode completion after admission) |
| ttft_ns / latency_ns / queuing_delay_ns / tpot | measured (same definitions as the original CSV) |
| itl_count / itl_mean_ns / itl_p50_ns / itl_p99_ns / itl_max_ns | measured (from the request's ITL list) |

Full ITL lists live in the side file `request_itls.jsonl`
(`{request_id, itl_ns}`), since a CSV cell cannot hold a list.

## Rolling window aggregates

Computed offline from the two files (see
`experiments/fault_detection/ml/features.py:build_windows`); window
length configurable, each window stamped with window-end sim time.

## Ground truth (NEVER in observable files)

`injection_audit.jsonl` + manifest only. A loader test must fail
if an observable file contains manifest-only columns.
