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

## Submission-readiness pass (host only, no sim runs)

- Verified from committed-equivalent artifacts with
  `verify_smoke.py` (Windows TEMP, not committed): 40 windows (22
  healthy, 18 anomalous); whole-set tallies fixed TP=6 FP=0 FN=12
  TN=22, residual TP=7 FP=1 FN=11 TN=21; split arithmetic
  cross-checks on all six metrics.csv rows (test = 9 anomalous + 8
  healthy; train = 9 + 14); thresholds fixed=16646350ns,
  residual=424168ns. Healthy-heavy decode windows read 11–17 ms with
  residuals near zero; faulty-light decode windows read ~22 ms with
  residuals ≈ +11 ms and peer ratios 1.8–2.3; faulty-run prefill
  windows carry no mechanism effect (label-construction ceiling on
  recall). No rerun of the smoke matrix: artifacts postdate all
  behavior-changing commits and sims are deterministic.
- Committed reproducibility bundle `experiments/fault_detection/smoke/`
  (23 files, ~491 KB): README, manifest.jsonl, metrics.csv, verified
  report.md, architecture.svg, per-pair workload + telemetry + audit
  under `data/`. Raw run dirs (inputs/, logs, sim CSVs, ~30 MB) stay
  gitignored. `experiments/fault_detection/README.md` added
  (contribution table, scope notes).
- `PROGRESS_REPORT.md` reconciled: statuses rewritten to
  COMPLETED/ONGOING/PENDING, detector sketch matches implemented
  baselines, localisation stub disclosed, no MLP/GBM/macro-F1/
  confusion/top-k/mechanism-task claims, related work reduced to
  repo-present citations, metrics line qualified, HW-counter and
  profiled-not-trained statements added, diagram fixed with SVG
  reference. `fault_injection/README.md` stale audit path corrected.
- `python -m pytest tests/ -q` -> 21 passed on host.

## Synthetic training study (host only, NOT simulator output)

- Resampled the 40 real smoke windows (22 healthy / 18 anomalous,
  18-dim observable features) with 0.15-sigma Gaussian jitter,
  seed 42: train 1000 (100 fail-slow, 10%), val 200 (20), test 300
  (30). Scripts in Windows TEMP (`ground_stats.py`, `synth_study.py`),
  not committed. Baselines reuse the run.py protocol (99th pct of
  healthy TRAIN): fixed=16959038ns, residual=559103ns. Logistic =
  manual minibatch numpy, batch 32, lr 1e-4, weighted BCE (pos x9),
  standardized features, 15 epochs as requested.
- Test result: fixed P 0.600 / R 0.200 F1 0.300 FPR 0.015;
  residual P 0.833 / R 0.333 F1 0.476 FPR 0.007;
  logistic P 0.106 / R 0.367 F1 0.164 FPR 0.344.
- Diagnostic rerun at 200 epochs (same seed/config): loss still
  falling (1.2455 -> 1.0502), test F1 only 0.200, FPR 0.322.
  Verdict: the requested lr/epoch budget underfits (465 steps barely
  move the weights) while pos_weight x9 inflates the intercept, so the
  model over-fires on healthy windows. Synthetic resample does not
  validate real-hardware detection; kept out of the repo.
- v2 (train/test only, no val): same generator, train 1000 (100),
  test 300 (30). Logistic = StandardScaler (train-fit, constant cols
  5,6,12,17 dropped), sklearn lbfgs, balanced weights, C=0.1 by
  5-fold stratified CV on train (cv-F1 0.559), threshold frozen 0.5,
  converged in 21 iters. Test: fixed P 0.545 / R 0.200 F1 0.293;
  residual P 0.833 / R 0.333 F1 0.476 FPR 0.007;
  logistic P 0.395 / R 0.567 F1 0.466 FPR 0.096 (best recall,
  within 0.01 F1 of residual as predicted). Confirms v1 gap was
  optimization budget, not signal; residual remains best F1/FPR.
- v3 (proper weight x C grid, max_iter=20000): 5-fold CV on train
  over C {0.01..1.0} x weights {balanced, 1:5, 1:3}; chose 1:5/C=0.3
  (cv-F1 0.673), converged in 26 iters, threshold 0.5. Test:
  fixed F1 0.390; residual F1 0.565 FPR 0.011;
  logistic P 0.720 / R 0.600 F1 0.655 FPR 0.026 (18/7/12/263) —
  first config to beat residual, via weighting not capacity.
  Plot + metrics CSV in <Project>/plots/ (user-requested; not part
  of the repo tree).
- v4 training curves (same data/config 1:5/C=0.3): lbfgs loss-vs-iter
  via warm-started single steps — 54 iters, train log-loss
  0.5930 -> 0.2838, train F1 0.306 -> 0.667; test reproduces v3
  exactly (F1 0.655). Minibatch SGD rerun (bs=32, lr=0.1, 100 ep,
  lam=1/(C*n)): wBCE 0.5562 -> 0.4086, train F1 to 0.663, test F1
  0.667 / FPR 0.022 — matches lbfgs within noise. Curve PNG in
  <Project>/plots/synth_v4_training_curve.png. (max_iter=1
  ConvergenceWarnings during instrumentation are expected noise.)
- v5 pure focal loss, no L2 (gamma=2, grad-checked rel-err 2.6e-10):
  3-fold CV on train-1k over alpha {0.25, 0.5} x batch {8,16,32},
  lr linear-scaled (0.1@bs32). Batch size did not differentiate
  within noise (identical CV-F1 across bs per alpha); alpha=0.5 won
  (0.654 vs 0.523 — mild 1:9 imbalance wants focusing, not positive
  down-weighting). Winner bs=8 by tiebreak. Stage A test: focal
  P 1.000 / R 0.533 F1 0.696 FPR 0.000 vs residual 0.723, fixed
  0.545. Stage B 10x (train 10k/test 3k, same winner): focal F1
  0.681 FPR 0.000 (155/0/145/2700) vs residual 0.589, fixed 0.509 —
  learned model held steady while percentile rules degraded on the
  larger draw. NOTE: first v5 run was voided by a double-shuffle bug
  (X/y permuted independently — all detectors ~0, caught by the
  baseline sanity check); fixed to single permutation, reran.
  Plot + CSV in <Project>/plots/synth_v5_focal.*.
- v6 alpha grid {0.4, 0.5, 0.6} (bs=8, gamma=2, no L2, thr 0.5,
  1k stage only): CV-F1 0.569 / 0.578 / 0.588 (monotone but gaps <
  fold-std ~0.07 — ordering suggestive, not significant). Test:
  a0.4 F1 0.500 (10/0/20/270); a0.5 = a0.6 F1 0.571 (12/0/18/270),
  all P 1.000 FPR 0.000 vs residual 0.565, fixed 0.390. The 0.5->0.6
  tilt moved no extra test window across the frozen 0.5 line
  (prediction of recall-up/precision-down wrong in detail — discrete
  threshold saturation, not a smooth tradeoff). Plot + CSV in
  <Project>/plots/synth_v6_alpha.*. (One voided v6 run: CSV-name
  parse bug crashed plotting after metrics printed; fixed, reran —
  training numbers deterministic-identical.)

## Day-simulator build (M0-M6, host + container status as noted)

Scope: resource-aware multi-GPU day simulator on Qwen3-30B-A3B (MoE 128/top-8),
RTX PRO 6000, TP=2, PP=2 (existing `single_node_moe_pp_instance.json` shape;
new P/D variant `configs/cluster/moe_pd_tp2_pp2_instance.json`). Full branch
status at start: HEAD e691da5, `M docs/fault_injection/RUNLOG.md` (prior
synthetic-study notes, preserved), untracked prior-session scaffolding
(`day_workload.py`, `gpu_state.py`, tests, moe P/D config) adopted, not reset.

- `git rev-parse HEAD; git status --short` -> e691da5, main; see file list above.
- `docker ps` -> FAILED: `failed to connect to the docker API at
  npipe:////./pipe/dockerDesktopLinuxEngine ... The system cannot find the
  file specified.` Docker daemon down on this host. Consequence: NO container
  sims possible in this session (no ASTRA-Sim binary on host). All sim runs
  (M1 smoke, M6 sim comparisons, `validate.sh`) are BLOCKED and recorded as
  such; host-side verification only.
- `pytest tests/test_day_workload.py tests/test_gpu_throttle.py tests/test_fault_injection.py tests/test_fault_ml.py -q` -> 31 passed (host, 3.32s).
- `python -m experiments.fault_detection.day_workload --out $env:TEMP\day_smoke.jsonl --horizon-h 0.25 ...` -> `wrote 354 rows (0.25h horizon)`; first rows valid JSONL schema.
- Router round-trip FAILED first: `Router.load_requests` with absolute TEMP
  path -> `OSError: [Errno 22] Invalid argument: '"'"'../C:\\Users\\...'"'"'`.
  Root cause: `serving/core/router.py:119` unconditionally prefixed `"../"`
  (runtime chdir convention), breaking absolute paths. Fix: `os.path.isabs`
  guard (same pattern as fault/telemetry paths in `__main__.py:457,463`);
  relative behaviour unchanged. Regression test `tests/test_router_paths.py`
  (absolute + relative-dotdot). Reran both forms: absolute 354 reqs OK,
  relative 5 reqs (req_num cap) OK.
- `pytest tests/ -q` -> 33 passed (incl. 2 new router tests).
- M1 static verification (host): `_resolve_parallelism` on Qwen3-30B-A3B
  (48 layers, 128 experts): moe_pd_tp2_pp2 -> prefill tp=2 pp=2 ep=2 n=4,
  decode same (total 8 instance-GPUs; cluster total_npu=12 with prefill
  shadow range); `single_node_moe_pp_instance` -> tp=2 pp=2 ep=2;
  `single_node_moe_single_instance` -> tp=2 pp=1 ep=2. EP=TP<=TP, 128%2==0,
  pp<=48 all pass. Profile check: tp1+tp2 dense/attention/per-seq/skew
  present; moe.csv tp1-only (analytic per-EP-rank lookup, documented).
  Deterministic container smoke BLOCKED (docker, above).
- M5 wiring: 7 edits in `serving/__main__.py` (flags, init+validation,
  tick, busy accounting, factor hook, audit write) + `_throttle_tick`
  helper; `python -m py_compile` COMPILE-OK; `python -m serving --help`
  shows `--throttle-config/--throttle-audit-output`.
- Validator fix: example config `_comment` key rejected -> allow
  `_`-prefixed metadata keys; `load_throttle_config` on
  `example_throttle/decode_pool_moe_pd_tp2_pp2.json` -> gpu_ids [8..11] OK.
- `pytest tests/ -q` -> 36 passed (incl. compute-only scaling, tick cap).
- M4 workload stats (host, seed 7, base 2.0): 24h gen 143034 reqs in
  0.60s wall; peak-hour 23144 vs night 3166 (7.31x); rate noon 8.0 vs
  3am 0.8; peak disabled -> noon-hour 6152; peak shifted 15h -> max
  hour 14; repro True.
- M6 state-model study (TEMP script, NOT simulator output):
  1 steady u=0.5 -> L=0.500, 0 transitions, never throttled;
  2 noon-peak thresholds 1.0/0.95 -> peakL 0.881, 0 transitions;
  3 noon-peak defaults -> 2 transitions (on+off), finalL 0.002 (recovery);
  4 forced-moderate (th 0.35/0.25) -> throttled, 1 transition;
  5 prefill-hot/decode-cool -> prefill 2 transitions, decode 0, load 0.2
  (isolation holds). Two voided runs: tuple-unpack bug (transitions are
  3-tuples) and a stale summary label; fixed, reran.
- PP=4 MoE: no config exists and no profile/topology validation run;
  recorded as UNVERIFIED (not attempted).

## 2P2D TP=2 Qwen3-32B topology (primary, host validation only)

- New `configs/cluster/qwen32_2p2d_tp2.json`: 2 prefill + 2 decode stations,
  TP=2 each (8 GPUs), Qwen3-32B dense on RTX PRO 6000, link_bw 16 /
  link_latency 20000 (repo-calibrated defaults, kept per user approval --
  not PCIe-spec values). Per-pool budgets mirror single_node_heterogeneous.
- Host check `_resolve_parallelism` x4 -> tp=2 pp=1 ep=1 num_npus=2 each;
  total_npu=12 (4+4shadow+4). tp2 profile bundle complete (5/5 files).
  Memory arithmetic: 64GB weights -> 32GB/rank on 96GB cards, fits.
  64 heads / TP=2 = 32/rank, head-safe.
- `pytest tests/ -q` -> 36 passed (host).
- BLOCKED (docker daemon down, re-verified this session): routing proof
  (both stations serve), deterministic 2x smoke, link_bw sensitivity
  sweep, `validate.sh`. Next: restart Docker, 10-req smoke x2 identical
  seeds -> identical Total clocks; telemetry instance_id split nonzero on
  all 4 instances; then sensitivity 16/32/64.
- Standing caveats recorded in config `_comment`: no hardware validation
  for this topology; P0->D0 vs P0->D1 transfer costs identical by
  construction (per-prefill-instance shadow-range modeling).

## E2E validation session (2026-10-12, Docker available after restart)

HEAD e691da5, branch main. Working tree preserved as found (prior RUNLOG
notes, validate.unix.sh untouched). No commits. Temp configs/outputs under
gitignored experiments/out/ only. Helper scripts summarize_run.py,
phase5_check.py live in experiments/out/ (ignored, not committed).

### Phase 1 -- environment (all commands exit 0 unless noted)
- `docker ps` initially FAILED (daemon down, npipe missing). Started
  `C:\Program Files\Docker\Docker\Docker Desktop.exe` (non-destructive),
  waited 60-90s: daemon up. Pre-existing exited container
  `servingsim_docker` (image astrasim/tutorial-micro2024) restarted with
  `docker start` (not recreated).
- ASTRA-Sim binary present:
  astra-sim/build/astra_analytical/build/AnalyticalAstra/bin/AnalyticalAstra.
  Container python3 imports yaml/pandas/numpy/rich/pyinstrument OK.
- Host `pytest tests/ -q` -> 36 passed (later 37/37 after regression test).
- Daemon died mid-session (process gone); restarted Docker Desktop again,
  `docker start servingsim_docker`; bind-mounted outputs intact.

### Phase 2 -- deterministic smoke (qwen32_2p2d_tp2, Qwen3-32B TP=2x2P2D)
- Workload: `python -m experiments.fault_detection.workloads --out
  experiments/out/e2e_smoke16.jsonl --num 16 --rate-sps 4.0 --seed 42
  --pattern poisson --short-in 64 --long-in 512 --short-out 32
  --long-out 256` -> 16 rows. (gitignored: verified via check-ignore.)
- r1/r2/r3 identical CLI (`--dtype bfloat16 --block-size 16 --num-reqs 16`,
  run-ids e2e_r1/r2/r3, telemetry dirs): all completed, 16/16 reqs.
- Total clocks: r2=r3=9509780493 ns. r1.csv==r2.csv and r2.csv==r3.csv
  byte-identical (diff -q silent). Determinism PROVEN.
- Station split (all runs): prefill 0:8/1:8, decode 2:8/3:8. All four
  stations serve. 1222 batch rows (P:18+18, D:574+612). p50 latency
  4.42s, p99 7.71s; TTFT p50 57.4ms.

### Phase 3 -- throttling-disabled control: PASS
- r1/r2/r3 ran without --throttle-config. Telemetry dirs contain exactly
  batch_telemetry.csv/request_telemetry.csv/request_itls.jsonl; no
  throttle_audit.jsonl. No --fault-config used (fault hooks untouched).

### Phase 4 -- causal validation (fixed trace e2e_smoke16.jsonl)
- DEFECT FOUND+FIXED: first thr_B run crashed at first window tick:
  `_throttle_tick` KeyError: 0 (`__main__.py:165`) -- tick iterated ALL
  instance GPUs but manager holds only configured ones. Fix: tick iterates
  manager.states via new gpu_to_instance map; unconfigured GPUs keep
  factor 1.0. Regression test test_tick_partial_config_ignores...
  added. Host suite 37/37. (Failing log preserved at
  experiments/out/e2e_B.log? No -- overwritten by successful rerun;
  traceback recorded here instead.)
- B (unreachable th 1.0/0.99, machinery on): clocks 9509780493 ==
  disabled EXACT. 36 window rows, maxL 0.977, 0 transitions, 0 throttled
  batches. Deterministic equivalence PROVEN.
- C (decode GPUs 8-11, th 0.35/0.2, 1.5x, 1s windows): clocks 12537453277
  (+31.8%). Engage t=2.0s all 4 GPUs. 473 throttled batches factor
  exactly 1.5 (applied/nominal=1.4999). Latency p50 4.42->5.70s (+29%),
  p99 7.71->10.55s (+37%). Batch mean 30.7M->40.3M ns (+31%). TTFT
  p50/p99 IDENTICAL (57.4M/93.0M) -- prefill isolation PROVEN.
- C2 (same, 2.0x): clocks 15594675002 (+64.0%). p50 7.06s. Monotone
  magnitude 1.0x/1.5x/2.0x -> 9.51/12.54/15.59s. Ratio exact 2.0000.
- D (burst-gap-burst 35-req day_workload trace, thr_C): 105.4s sim.
  Transitions on@18 -> off@31/40 -> on@46 -> off@52 -> on@70/72 ->
  off@82/85 -> on@95 -> off@100 -> on@105(end). Full hysteresis cycling,
  per-instance independence (D0 off@31 vs D1 off@40). Recovery PROVEN.
  1085 throttled batches @1.5x exact.
- Check 10: ml/features.py untouched (git status clean for it);
  batch/request telemetry headers contain no throttle columns.

### Phase 5 -- workload integration: PASS
- poisson/bursty/sustained/day_workload CLI all generate (5/5/5/21 rows).
- phase5_check.py: workload arrivals == telemetry arrival_ns EXACT
  (16.6s..103.3s, monotonic), burst2 cohort (15 reqs after t=60s)
  present, last completion 105.4s > last arrival 103.3s (drain observed).
  Sim consumed generated timestamps verbatim.

### Phase 6 -- link_bw sweep (temp configs, 16-req trace): DONE
- bw16/32/64 clocks: 9509780493 / 9445550736 (-0.68%) / 9413781388
  (-1.01%). Same 1222 batches + routing; TTFT p50 57.4M->48.0M->43.2M
  (transfer on prefill critical path, responds to BW); batch max
  93.0M->74.8M->65.6M. Compute dominates; comm partially on critical
  path. No PCIe/NVLink claims (ring-formula abstraction, per plan).

### Phase 7 -- regression
- Host suite final: 37/37 pass.
- `validate.sh --clocks-only`: FIRST attempt failed environmentally
  (ran stripped copy from /tmp -> REPO_ROOT misresolved to /tmp:
  `No module named serving`). SECOND attempt ran full suite: 62 FAIL
  lines, ALL with +0.0000% delta, 0 nonzero, 0 unfinished -- root cause:
  CRLF working copy makes awk $2 carry \r so string compare fails
  (baselines file is CRLF; AGENTS.md documents this pitfall). Spot
  re-run with CR-stripped baselines copy (BASELINES=/tmp/..., zero repo
  changes): single/pd/pd_fault_decode_mult all PASS, incl. fault
  baseline 2318421829 == DESIGN.md recorded value (fault hooks
  untouched). Temp `serving/.validate_tmp.sh` removed afterwards.
- Nothing staged: git status shows only intended source/doc/test files;
  all sim outputs + temp configs under ignored experiments/out/.

## Dataset + demo showcase session (2026-10-12, container, Qwen3-32B 2P2D TP=2)

Paired 16-req dataset (identical trace e2e_smoke16.jsonl, seed 42):
- A healthy (e2e_r2): clocks 9509780493.
- B unreachable-threshold: identical clocks (prior session).
- C throttled 1.5x decode (thr_C.json): clocks 12537453277 (+31.8%).
- C2 throttled 2.0x (thr_C2.json): clocks 15594675002 (+64.0%).
- F faulty decode 2x both stations (experiments/out/demo_fault.json --
  NOTE: first attempt failed validation on a _comment key (strict
  fault_injection validator); removed the key, no code change):
  clocks 15559896675 (+63.6%), 523 fault audit rows, p50 7.78s,
  TTFT p50 66.4ms (backpressure vs 57.4 healthy).
- FT faulty + throttled 1.5x: clocks 21665154168 (+127.8%), p50
  10.83s, p99 19.68s, 523 fault + 571 throttle audit rows, 487
  throttled batches @1.5 exact, GPUs 8-11 mean_busy ~96.7%,
  throttled 20/21 windows. Effects compound (21.7 > 15.6, 12.5).
Daily demo (demo_day.jsonl, 71 reqs/30min, seed 7): healthy run
demo_H clocks 1773499805661 (29.6 sim-min), wall ~2.5h. Telemetry:
p50 latency 4.5s, TTFT p50 72ms, prefill busy 0.3%, decode ~19% each,
peak minute 5 req, max queue depth 2. Light load: daily SHAPE shown,
no queueing pressure at this volume (documented, not a failure).
Cost datum: 71 heavy reqs ~= 2.5h wall; full-volume 24h sims are
wall-clock-prohibitive -- scaled/compressed demos required.

## Consolidation + instrument-repair session (2026-10-25, host + container)

The instrument was reviewed, the measurement defects fixed, and the code
consolidated from 15 modules into 4 + 1 notebook. Commands and outputs
below are the executed record; `report.md` §16 is the written account.

### 1. Host unit tests

```
python -m pytest tests\ -q
-> 90 passed in 3.94s
```

(88 after the module merge; +2 in this session:
`test_probe_run_ids_are_disjoint_and_path_safe` and
`test_calibrate_stays_inside_its_explicit_bracket`, plus the alpha cases
added to `test_config_validation`.)

### 2. Smoke recheck: the five frozen clocks still hold

```
python -u -m experiments.fault_detection.inject_faults \
  --matrix experiments/fault_detection/matrix.yaml --profile smoke \
  --out experiments/out/smoke_recheck2
```

| run | clocks (ns) | applied | U_decode |
|---|---|---|---|
| light_poisson healthy | 7667807319 | False | 0.541 |
| light_poisson faulty (decode x2) | 8387062877 | True | 0.824 |
| heavy_poisson healthy | 4862442514 | False | 0.995 |
| bursty healthy | 8794093936 | False | 0.997 |
| bursty faulty (handoff 100 ms) | 8894547812 | True | 0.997 |

All five are **bit-identical** to the frozen bundle in
`experiments/fault_detection/smoke/data/manifest.jsonl`; `0 failed run(s)`.
This is the check that the refactor changed no simulator behaviour.

The first attempt (`experiments/out/smoke_recheck`) failed it and found a
real defect: `_build_workload` did `params.pop("pattern", "poisson")`, so
every `bursty` workload was generated as Poisson. Symptom: bursty healthy
clocks `4862442514` == heavy healthy clocks `4862442514` (the bursty run
was silently the heavy run). Fixed by keeping `pattern` in the params and
pinned by `test_build_workload_keeps_the_requested_pattern`.

### 3. Re-evaluating the frozen smoke bundle with the repaired instrument

```
python -m experiments.fault_detection.evaluate \
  --data experiments/fault_detection/smoke/data \
  --report %TEMP%\smoke_new
-> wrote metrics.csv, report.md and windows.csv (33 windows)
```

| | pre-fix (frozen `smoke/metrics.csv`) | post-fix |
|---|---|---|
| windows | 40 (22 "anomalous") | 33 (8 root_cause, 25 unaffected, 7 downstream excluded) |
| fixed threshold, test | P 1.000 / R 0.556 | P 1.000 / R 1.000 (5/5), FPR 0.000 |
| residual, test | P 0.857 / R 0.667 | P 1.000 / R 1.000 (5/5), FPR 0.000 |
| localisation | 0.475 (stub that echoed ground truth) | top-1 0.875, coverage 0.625, selective top-1 1.000 |
| detection delay | 1.0 s | 1.5 s mean (0.5 s stride) |

Read this as a **label/feature fix, not a performance gain**: the old run
labelled every window inside the schedule window anomalous (including
downstream windows where the target instance was not the slow one), and
the old localiser copied the ground-truth label. Recall is now computed
over 8 root-cause windows instead of 18 mixed ones, and the confidence
interval is wide (recall CI [0.566, 1.000] with 1 test run). The frozen
bundle stays in the repo as the clock baseline; its `metrics.csv` is
historical.

### 4. Load calibration, 2P2D tp=1, Llama-3.1-8B

```
python -u -m experiments.fault_detection.run_simulation calibrate \
  --cluster configs/cluster/pd_2p2d_small.json --target-u 0.40 \
  --out experiments/out/pilot_2p2d_u40 --num-reqs 20 --iters 4 \
  --tol 0.08 --seed 11
```

| probe | lambda (req/s) | U_decode | U_prefill | TPOT p50 | span |
|---|---|---|---|---|---|
| 0 | 1.789 | 0.837 | 0.012 | 11.36 ms | 13.5 s |
| 1 | 0.299 | 0.230 | 0.002 | 11.27 ms | 74.4 s |
| 2 | 0.731 | 0.507 | 0.005 | 11.28 ms | 31.0 s |
| 3 | 0.468 | 0.346 | 0.003 | 11.28 ms | 47.9 s |

chosen `lambda=0.468/s (U_decode=0.346, target 0.40)`. Two facts worth
recording: (a) decode utilisation is far from linear in lambda and
saturates quickly, so the matrix's original guess of 0.78 req/s for
U=0.40 was ~0.2 too high; (b) U is measured per run, and the same lambda
gave U_decode 0.688 at 40 requests vs 0.837 at 20, so a regime is only
meaningful with its request count and span attached.

### 5. Defect found while calibrating: shared ASTRA-Sim input tree

Two calibration processes were started against the same out-root (one
was thought to have died -- it had not; a PowerShell-mangled `grep -E
"a|b"` had turned into a pipe and hid it). Both used simulator
`run_id="probe_01"`, and `build_run_paths` places the ASTRA-Sim input
tree at the *shared* `astra-sim/inputs/runs/<run_id>`, so each run
rewrote the other's workload while it was being read. Nothing crashed and
no warning was printed: the failure mode is a silently mixed trace.

Fixed by `probe_run_id(out_root, tag)`, which puts the out-root basename
in the run id (`pilot_2p2d_u40_probe_01`), and pinned by
`test_probe_run_ids_are_disjoint_and_path_safe`. The calibration CLI also
gained `--lambda-lo/--lambda-hi`, because the default two-decade bracket
spends probes above saturation where U is flat (pinned by
`test_calibrate_stays_inside_its_explicit_bracket`).

### 6. Load calibration, 2P2D tp=1, target U_decode 0.75

```
python -u -m experiments.fault_detection.run_simulation calibrate \
  --cluster configs/cluster/pd_2p2d_small.json --target-u 0.75 \
  --out experiments/out/pilot_2p2d_u75 --num-reqs 44 --iters 2 \
  --lambda-lo 0.5 --lambda-hi 3.0 --tol 0.08 --seed 12
```

| probe | lambda (req/s) | U_decode | U_prefill | TPOT p50 | span |
|---|---|---|---|---|---|
| 0 | 2.191 | 0.837 | 0.018 | 11.4 ms | 25.0 s |
| 1 | 1.621 | 0.723 | 0.016 | 11.4 ms | 27.1 s |

chosen `lambda=1.621/s (U_decode=0.723, target 0.75)`, 44 requests.
Measured again inside the matrix, the same lambda gave U_decode 0.849,
0.897 and 0.929 on three different seeds: at 44 requests U_decode varies
+/-0.13 run to run, so the matrix regime table reports the median with
the run count attached and never a single-run value.

### 7. Parallel workers: proved safe before using them

`--workers 3` was used for the matrix, which is only defensible if the
workers cannot touch each other's files. The smoke profile was re-run
three ways and the five clocks compared bit for bit:

```
python -m experiments.fault_detection.inject_faults \
  --matrix experiments/fault_detection/matrix.yaml \
  --profile smoke --out /tmp/smoke_par --workers 3
```

| run | sequential clocks (ns) | 3 workers (ns) |
|---|---|---|
| light_poisson healthy | 7667807319 | 7667807319 |
| light_poisson faulty | 8387062877 | 8387062877 |
| heavy_poisson healthy | 4862442514 | 4862442514 |
| bursty healthy | 8794093936 | 8794093936 |
| bursty faulty | 8894547812 | 8894547812 |

Identical. The workers are safe because every run id is disjoint (the
pair id contains the seed and the noise level) and each run owns its own
`inputs_*` tree; the shared ASTRA-Sim input tree defect from item 5 was
what made this unsafe before.

Three workers, not more: `--workers 6` was tried on the same smoke
profile and never finished. One simulator sat at 99.9% CPU for 11
minutes on a 10-request run that takes ~40 s, so the attempt was killed
and `/tmp/smoke_w6` removed. The cause was not diagnosed, so the matrix
runs at the proven level of 3 and no throughput claim is made beyond it.
At 3 workers the container uses ~55% of one core, i.e. the run is I/O
bound on the Windows bind mount, not CPU bound.

### 8. Three defects found while validating the matrix

All three were silent (no crash, no warning):

1. `_build_workload` popped `pattern` out of the workload dict before
   building it, so a `pattern: bursty` workload was generated as Poisson.
   The bursty load-matched negative was therefore not bursty. Fixed by
   keeping `pattern` in the dict; pinned by
   `test_build_workload_keeps_the_requested_pattern`.
2. The manifest was flushed once per PAIR, not per run. A pair is 1-12
   simulator runs (20-40 min), so one bad run at the end lost the whole
   pair's manifest lines and the work had to be re-simulated. Now flushed
   per run under a `threading.Lock`; pinned by
   `test_manifest_is_flushed_per_run_not_per_pair`.
3. The throttle template used `tau_s=20.0`, but the accumulator
   `L(t) = u*(1-exp(-t/tau))` cannot reach `th_on=0.55` inside a 20-25 s
   review run, so the throttle never engaged and the run was a silent
   neutral. `tau_s` is now 8.0; pinned by
   `test_throttle_template_can_fire_inside_a_review_run`. `alpha` is also
   validated to `(0, 1]` instead of being trusted.

### 9. Defect: `evaluate` silently dropped every faulty run of the matrix

First evaluation of the partial `review_min` matrix returned 201 windows,
ALL labelled `unaffected`, `n_root_cause=0`, although the audits showed
`applied: true, instance: 3` on 1467 batches.

Cause: `inject_faults` names a twin's telemetry directory after its
suffix, and a pair with several fault variants writes `faulty0_telemetry`
... `faulty5_telemetry`; `evaluate.collect_dataset` derived the directory
from `fault_present` and looked for `faulty_telemetry`. That directory
does not exist, so every faulty run hit a bare `continue` and only the
healthy twins were scored. The result looked like "the detector found
nothing" rather than "the instrument read nothing".

Fixes: the manifest now records `telemetry_dir` per run; `evaluate`
resolves it from the manifest, falls back to the `run_id` tail (which is
exactly the directory stem, so bundles written before the field existed
still resolve to their OWN twin rather than to twin 0), then to the
healthy/faulty default; and a run whose telemetry cannot be found is
reported (`skipped_runs`, `n_runs_missing_telemetry`) instead of being
dropped. Pinned by
`test_multi_variant_pair_resolves_each_telemetry_dir`,
`test_legacy_manifest_without_telemetry_dir_still_resolves`,
`test_legacy_multi_variant_manifest_resolves_by_run_id` and
`test_missing_telemetry_dir_is_reported_not_silently_dropped`.

Also fixed while re-reading the report writer: the bootstrap section
printed its heading whenever ANY row had a bootstrap interval, so a
single-run test split printed "Bootstrap over runs" with no rows under
it.

### 10. Finding: `kv_transfer_bandwidth` carries no latency signal

The three transfer-bandwidth severities of `review_u40_small_2p2d_1`
produce bit-identical total clocks:

| run | bandwidth_fraction | total clocks (ns) | U_decode |
|---|---|---|---|
| healthy | - | 43040878835 | 0.387 |
| faulty3 | 0.80 | 43043305415 | 0.387 |
| faulty4 | 0.67 | 43043305415 | 0.387 |
| faulty5 | 0.50 | 43043305415 | 0.387 |

The mechanism does change the trace -- the audit reports
`nominal_comm_sum` 10878976 scaled to 13598720 / 16237280 / 21757952,
i.e. exactly 1.25x / 1.5x / 2.0x, `applied=true` on all 9 audited
batches -- and the simulation still lands on the same nanosecond for all
three. There is no dose-response at this operating point, so there is
nothing for a detector to find; the family is an honest negative, not a
detector failure.

Consequence for the metric: scoring these windows as positives would
count a guaranteed miss against every detector, so `evaluate` now
classifies a fault family as timing-neutral when (a) two or more
severities give bit-identical total clocks, or (b) a single severity
reproduces the healthy twin's clocks exactly, and excludes its windows as
`neutral` (reported, never silently). Missing clocks never imply
neutrality. Pinned by
`test_timing_neutral_mechanism_is_excluded_not_scored`,
`test_dose_response_keeps_a_family_scoreable` and
`test_fault_that_moves_clocks_stays_a_positive`.

### 11. Regression after the two fixes

Host suite and the frozen smoke bundle were re-run after items 9 and 10:

```
python -m pytest tests\ -q
99 passed in 15.81s

python -m experiments.fault_detection.evaluate \
  --data experiments/fault_detection/smoke/data --report %TEMP%\smoke_recheck
-> wrote metrics.csv, report.md and windows.csv (33 windows)
```

`n_windows=33`, `n_root_cause=8`, `n_unaffected=25`,
`n_excluded_downstream=7`, `n_runs_missing_telemetry=0`,
`n_timing_neutral_families=0`, `localization_top1=0.875` -- identical to
the post-repair numbers in item 3, so neither fix changed the frozen
bundle. `n_timing_neutral_families=0` is the right answer there: the
smoke twins have different clocks from each other and from healthy, so
no family is neutral.

### 12. The calibrated matrix completed (100 runs, 0 failed)

```
python -u -m experiments.fault_detection.inject_faults \
  --matrix experiments/fault_detection/matrix.yaml \
  --profile review_min --out experiments/out/review_min \
  --seeds 1,2,3,4,5 --workers 3
...
done, 0 failed run(s)
```

100 manifest lines: 15 healthy twins + 85 faulty twins (5 seeds x
(u40 7 twins + u75 11 twins + bursty 1 twin)). Wall clock ~85 min at 3
workers, i.e. ~1.1 runs/min, I/O bound on the Windows bind mount. The
evaluation that produced `report.md` §16.5:

```
python -m experiments.fault_detection.evaluate \
  --data experiments/out/review_min --report experiments/report/review_min
-> wrote metrics.csv, report.md and windows.csv (2542 windows)

n_windows=2542  n_root_cause=676  n_unaffected=1866
n_excluded_downstream=2379  n_excluded_recovering=1  n_excluded_neutral=554
n_runs_missing_telemetry=0  n_runs_missing_audit=0
```

### 13. Three defects found by auditing the report against its own dump

All three were silent: the report looked self-consistent, and only
reading the dump against the prose exposed them.

(a) **`windows.csv` held only the scored rows.** The header reported
"excluded: downstream 2379, neutral 554" while the dump contained zero
rows of either kind, so no exclusion could be checked and 554 looked like
it was one family. The dump now writes every row: `unaffected` 1866,
`downstream` 2379, `root_cause` 676, `neutral` 554, `recovering` 1 =
5476, with metrics still computed on the 2542 scored rows. Reading it
also corrected the prose: the 554 are 254 from the `network` twin (audit
`applied=false`) plus 300 from the timing-neutral `transfer_bw` twins.
Pinned by `test_windows_dump_contains_the_excluded_kinds`.

(b) **Localisation compared a link label with NPU instance ids.**
`kv-transfer-network:0` names a transfer link; NPU instance 0 is the
prefill. Every handoff window whose top-ranked instance happened to be 0
was scored as a correct localisation: 63 of 177. Reported top-1 0.831
instead of 1.000 over the 499 NPU-labelled positives; the 177 link
positives are now reported as `localization_not_localisable`, because a
link is not in the candidate set. Pinned by
`test_link_label_is_never_a_localisation_hit` and
`test_npu_label_is_still_scored_after_the_link_split`.

(c) **The delay path applied `min(thresholds)` to every role.** The
decode window threshold is 13,650 ns; prefill healthy residuals are
~4.0M ns. Prefill crossed the decode threshold in every run, so all 85
faulty runs "alarmed" (mean 0.82 s) -- including the 5 `network` runs
whose audit says `applied=false` and the 30 timing-neutral `transfer_bw`
runs. The statistic is now calibrated on healthy runs at its own
granularity: 73 of 85 alarm, `decode_slow` 30/30 (mean 1.60 s),
`throttle` 5/5 (mean 8.50 s, consistent with `tau_s=8`), but also
handoff 10/15, `network` 4/5, `transfer_bw` 24/30. So the delay number is
an upper bound and the latency axis is not established (report §16.7 item
13). Pinned by
`test_delay_does_not_alarm_when_the_fault_was_never_applied`.

Host suite after the three fixes: `107 passed in 3.34s`.

### 14. Honest negatives from the matrix, with exact clocks

- **`kv_transfer_bandwidth` is timing-neutral.** In
  `review_u40_small_2p2d_1`, severities 0.8 / 0.67 / 0.5 all give
  `43043305415` ns, while the audit records comm size x1.25 / x1.5 / x2.0.
  The trace changed and the simulation did not.
- **Handoff is negligible and non-monotonic.** Same pair, healthy
  `26551433758` ns: 20 ms -> +10,816,476 ns (+0.041%), 50 ms ->
  +7,013,835 ns (+0.026%), 100 ms -> +11,143,874 ns (+0.042%); median
  decode batch duration 11.29 -> 11.32 ms. A 50 ms hold moves the clock
  less than a 20 ms hold.
- **The `network` twin injected nothing.** 5 of 5 runs carry
  `successfully_applied=false` with the manifest reason "audit
  inconsistency: records for network are inside the window but
  applied=false"; their clocks equal the timing-neutral `transfer_bw`
  twins exactly (`26552880486` ns). 254 neutral + 37 unaffected windows,
  no positives: an unexercised cell, not a negative result.
- **Smoke bundle after the localisation fix:** 33 windows (8
  `root_cause`, 25 `unaffected`, 7 `downstream` excluded),
  `localization_top1=1.000` over 5 NPU positives with 3 link positives
  reported as not localisable. The 0.875 in items 3 and 11 was the same
  link-label comparison defect, not a change in the data.

### 15. Final verification pass after the three fixes

Re-ran the whole instrument on the committed matrix, then the suite:

```
python -m experiments.fault_detection.evaluate \
  --data experiments/out/review_min --report experiments/report/review_min
-> wrote metrics.csv, report.md and windows.csv (2542 windows)

dump kinds: {'unaffected': 1866, 'downstream': 2379, 'root_cause': 676,
             'neutral': 554, 'recovering': 1} total 5476
localization_top1=1.000 [0.992, 1.000]  top3=1.000
localization_windows=499  localization_scored_positives=499
localization_not_localisable=177  localization_link_windows=177
localization_coverage=1.000  localization_selective_top1=1.000
mean_detection_delay_s=7.990  delay_runs_detected=73

python -m pytest tests\ -q
107 passed in 3.19s
```

The delay block in the generated report now carries a per-family table
(faulty runs, audit-applied runs, timing-neutral runs, runs with an alarm,
mean delay) plus the sentence that a crossing is not attributable to the
fault, naming the seven families whose crossings are unexplained. That is
what makes the `mean_detection_delay_s` number non-quotable as a detection
latency without the reader having to reconstruct the table.

Smoke bundle re-checked with the same code: 33 windows, 8 `root_cause`,
25 `unaffected`, 7 `downstream`, `localization_top1=1.000` over 5 NPU
positives, 3 link positives reported as not localisable.
