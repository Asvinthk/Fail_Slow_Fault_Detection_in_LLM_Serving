# Run log (commands + outputs, failures included)

## Stage 1 — inspect and baseline

- `git log -1 --oneline` -> `a4053bc Lead the README with a Latest News section`, branch `main`, submodule `astra-sim d346994`.
- `python -m serving --help` -> flags verified; `--fault-config`, `--fault-seed`, `--telemetry-output` added by this work.
- ASTRA-Sim binary was initially missing (Windows host). Docker Desktop
  4.78.0 was started, image `astrasim/tutorial-micro2024` pulled,
  container `servingsim_docker` created with repo mounted at
  `/app/LLMServingSim`, Python deps + Chakra installed, analytical
  backend built via `build/astra_analytical/build.sh` (100%, all
  targets). Binary confirmed at
  `astra-sim/build/astra_analytical/build/AnalyticalAstra/bin/AnalyticalAstra`.
  Note: Windows checkout uses CRLF; `build.sh` needed CR stripping
  inside the container to run (repo file untouched).
- PD baseline, twice (exit 0 both):
  `python3 -m serving --cluster-config configs/cluster/single_node_pd_instance.json --dtype bfloat16 --block-size 16 --dataset workloads/example_trace.jsonl --output outputs/fi_baseline_pd.csv --num-reqs 10 --run-id fi_baseline_pd`
  -> `Total clocks (ns): 1661784496`, CSV md5 `d88e5e4ac8d8e4e42c7dc683aacd1f3b` both runs, `cmp` identical. Determinism confirmed.
- `validate.sh` PD subset (`pd pd_sharegpt pd_per_instance_config moe_pd`,
  clocks stage): all 4 match recorded baselines exactly
  (pd 1661784496, pd_sharegpt 12851061128, moe_pd 2081236210,
  +0.0000%). Disabled-by-default holds. (The script prints a
  "changed" table only because the Windows CRLF baseline file leaves
  a trailing CR on the expected value; numbers are equal.)

## Stage 2 — fault runs (all exit 0)

- Empty faults `--fault-config .../empty.json` -> `1661784496`,
  CSV byte-identical to baseline (`cmp` clean). No-fault path untouched.
- `decode_slow_2x.json` (decode instance 1, factor 2.0, step 1-30 s):
  -> `Total clocks: 2318732875` (+39.5% vs healthy), audit 59 records,
  all `instance: 1`, comp sums doubled (e.g. 11092427 -> 22184854).
  Reran -> identical clocks (reproducible). Other instance audit empty.
- `transfer_half_bw.json` (transfer, fraction 0.5): -> `1661784496`
  (+0.0%), audit 1 record on prefill instance 0. Honest negative
  result: on `example_trace.jsonl` (2-22 token prompts) the KV bytes
  are so small that 2x inflation is invisible at ns resolution.
  Demonstrating transfer slowdown needs a transfer-heavy workload
  (long prompts); recorded as pending work, not claimed.
- Failures fixed along the way (recorded, not hidden):
  1. `--fault-config` relative path resolved from repo root while cwd
     is `astra-sim/` -> FileNotFoundError; fixed with `../` prefix
     (same convention as dataset/cluster paths).
  2. Trace rows are tuples -> `TypeError` on item assignment; fixed
     with copy-on-write in `apply_to_trace_rows` + regression test.
  3. Audit first written under `inputs_root` (auto-deleted) -> moved
     next to `--telemetry-output` / `--output`.

## Unit tests

- `python -m pytest tests/test_fault_injection.py -q` -> 8 passed on
  host Python 3.14 and container Python 3.10 (profiles, targeting,
  stall, validation errors, tuple rows).
