# Telemetry schema (observable only)

Enabled by `--telemetry-output <dir>`. The original `--output` CSV
is byte-identical with telemetry on or off.

## batch_telemetry.csv (OBSERVABLE)

Written per ASTRA-Sim iteration report.

| column | measured / estimated |
|---|---|
| run_id | measured (CLI) |
| instance_id | measured (NPU mapping) |
| batch_id | measured (ASTRA-Sim report id) |
| batch_time_ns | measured (report clock) |
| finish_ns | measured (same clock) |
| duration_ns | estimated in prototype (full per-batch duration needs scheduler pairing; currently 0 placeholder) |
| num_prefill / num_decode / total_len | estimated in prototype (needs Batch snapshot; currently 0) |
| com_cycle | measured (exposed communication cycles from `controller.parse_output`; simulator-instrumented proxy for transfer-exposed time, NOT a direct transfer-duration measurement) |

KV-transfer time is NOT directly separable from prefill iteration
time in the native simulator. Proxy used: exposed-comm cycles plus
the TTFT-vs-first-ITL split (handoff faults move ITL only).

## request_telemetry.csv (OBSERVABLE, planned)

request id, prefill/decode instance, input/output tokens, arrival,
first-scheduled, prefill-done (TTFT), decode-admit, first decode
completion, TTFT, TPOT, ITL stats + full ITL side file.
New fields go in `Request.__init__` (no `getattr` fallbacks).

## Ground truth (NEVER in observable files)

`injection_audit.jsonl` + manifest only. A loader test must fail
if an observable file contains manifest-only columns.
