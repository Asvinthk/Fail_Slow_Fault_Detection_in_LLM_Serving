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

## Stage 2-5 follow-up (host, Windows, no ASTRA-Sim binary)

- `python -m pytest tests/ -q` -> 18 passed (15 fault-injection +
  3 fault-ML: feature isolation guard, residual fit, observable-leak
  guard). Binary-dependent runs were not attempted on this host.
- Fault-semantics fixes (all covered by new unit tests):
  1. `compute_additive` was per-layer; now per-iteration (last serial
     row only). Multi-row test pins applied == nominal + delta.
  2. `kv_transfer_bandwidth` ignored the severity weight; now the
     effective factor is `1 + w(t)*(1/fraction - 1)` (ramp midpoint
     test: 800 B at w=0.5 -> 1200 B).
  3. `network_degradation` (mechanism) scaled comp_time as a placeholder;
     now scales `comm_size` on collective rows only, comp untouched.
  4. `network` targets now match by instance regardless of role.
- `python -m experiments.fault_detection.workloads --out <tmp>/w1.jsonl
  --num 5 --seed 7 --pattern bursty` (twice) -> md5 `3f28394d...`
  both runs. Seeded determinism holds.
- Matrix expansion check -> smoke 4 jobs, full 38 jobs.
- `python -m experiments.fault_detection.ml.run --data <empty> --report
  <empty>` -> clean message + exit 2 when no manifest exists
  (was FileNotFoundError traceback before the fix).
- New validate.sh scenarios (`pd_fault_decode_mult`,
  `pd_fault_transfer_bw`, `pd_fault_handoff`, `pd_fault_stall`) added;
  baselines recorded in-container (see container session below).

## Container session (servingsim_docker, ASTRA-Sim analytical binary)

All runs: `--dtype bfloat16 --block-size 16`, PD config
`single_node_pd_instance.json` unless noted, exit 0 throughout.
Headline: healthy 1661784496 ns (x3 identical runs, two CSV md5
`d88e5e4a...` identical); telemetry on/off identical clocks + CSV.

- decode_mult 1.5x -> 1989969195 (+19.8%), 59 audits.
- decode_mult 2.0x -> 2318421829 (+39.5%), 59 audits, per-batch comp
  exactly doubled (sums 656787414 -> 1313574828); x3 identical runs.
  Pre-fix record 2318732875 differs 0.013% (fault-boundary wake-ups;
  per-batch injection unchanged).
- additive +10 ms/iter -> 2251594628 (+35.5%), audit delta exactly
  10000000 ns on all 59 records (per-iteration, not per-layer).
- transfer_half_bw on short trace -> 1661594628 (-0.01%), 1 audit
  (655360 -> 1310720 B). Neutral: tiny KV hidden in iteration time.
- handoff_100ms from 1 s -> 1661127254 (-0.04%), 1 hold (prefill
  drains before onset; onset placement matters). Held request latency
  +110 ms, TTFT unchanged.
- handoff_100ms from 0 s -> 1761737862 (+6.0% = +100 ms makespan),
  9 holds; every latency +99.9..+111.2 ms; mean TTFT +4.38 ms
  (fault-active observer effect on batching, disclosed in DEVIATIONS).
- decode_stall 1-5 s -> 5657635463 (+3.996 s = the window), 3987
  skipped polls, 0 in-window batches, 10/10 requests (first attempt
  lost 1 request to silent drop in transfer_prefill_request; fixed to
  hold-and-retry, unit-tested, rerun confirms 10/10).
- network_1.5x on tp=1 -> 1661594628, 60 audits with zero changed
  bytes (no collective rows; now labeled applied=false, never silent).
- Onset-miss control: transfer fault over [1 s, 30 s] on a 0.4 s run
  -> clocks bit-identical, empty audit.
- Transfer calibration (1 req, 2048 prompt, 256 MB KV): link 16 GB/s
  healthy 255189132, 2x/4x identical (transfer overlaps sender
  compute, fully hidden); link 1 GB/s (`pd_1p1d_slowlink.json`)
  healthy 423217485 (+66%), 2x -> 673217485 (+250 ms),
  4x -> 1173217485 (+750 ms): linear in (1/fraction - 1).
- `./serving/validate.sh --clocks-only` (CR-stripped copy, LF
  baselines copy): 58/58 existing PASS, 4 new scenarios NEW with the
  values above; baselines recorded via `--update <name>` each.
  (Container note: working-copy CRLF breaks bash scripts and the
  baseline compare; strip CR in /tmp copies, never in the repo.)
- Telemetry fixes from measured data: per-report logging duplicated
  rows ~25x (ASTRA-Sim repeats finished lines) -> dedupe on
  (instance, sys, id); shape via report id - 1 mapping; stall windows
  emit skip-count summaries; healthy-run telemetry path prefix was
  nested under --fault-config (misplaced into astra-sim/; fixed,
  stray dir removed, data relocated, submodule clean).
- Smoke matrix (`generate --profile smoke`): 3 pairs, 5 runs,
  0 failed; faulty applied=True, healthy applied=False; clocks
  reproduce across invocations. Pair-grouping fix: healthy twin used
  to run twice per shared pair.
- ML report on smoke (40 windows): fixed-threshold test P 1.000 /
  R 0.556 FPR 0.000; residual-only P 0.857 / R 0.667 F1 0.750;
  logistic P 0.286 / R 0.222 (overfits smoke scale, honest negative);
  localization 0.475, abstention 0.075, delay 1.0 s (2 s floor).
