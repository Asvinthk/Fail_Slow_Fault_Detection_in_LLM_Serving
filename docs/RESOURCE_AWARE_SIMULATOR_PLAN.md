# Resource-Aware Multi-GPU Simulator Plan — Full-Day Load with Synthetic Throttling

> Status: PLAN ONLY — no implementation. No source, config, test, or output modified to produce this document.
> Conventions: **VERIFIED** = read in code at the cited path; **INFERENCE** = architectural reading; **PROPOSED** = new work.
> Effort letters (S/M/L/XL) are engineering estimates, not measured facts.

## 1. Executive summary

**VERIFIED.** LLMServingSim 2.0 (`casys-kaist/LLMServingSim`, local checkout HEAD `e691da5` on the `fail_slow` remote) is a usable base for this work: vLLM-V1 continuous batching, prefill/decode (P/D) disaggregation, profile-table compute timing + ASTRA-Sim analytical communication, deterministic clocks, instance/role-targeted fail-slow fault hooks (batch granularity, future-batches-only), observable-only telemetry with leak guards, paired seeded experiments, and a 58-scenario validation suite.

**Gap for this task.** The engine models **instances/workers, not independently-degrading GPUs**. There is one `hardware/model/tp` per instance and one latency per layer replicated to all ranks; `apply_to_trace_rows(faults, rows, role, instance_id, now_s)` cannot address a rank; in-flight batches cannot be rescaled; arrivals are stationary-rate (no diurnal peak); telemetry is per-instance with no utilization/load/throttle state.

**PROPOSED.** Keep the simulator; do not replace it. Add:

1. (a) Configurable P/D GPU pools mapped onto the existing instance abstraction (operating point: 1 GPU = 1 instance, `tp=1`), with an explicit per-rank extension point deferred to M4.
2. (b) A reproducible time-varying arrival generator emitting the unchanged JSONL schema, with a configurable noon peak.
3. (c) Per-GPU synthetic accumulated-load + threshold/hysteresis throttling applied **at schedule time to future batches only** (compute-only, never network).
4. (d) Minimal additive telemetry preserving the audit/manifest separation.
5. (e) Staged validation ending in healthy-peak vs degraded ablations that generate labelled, interpretable data for future fail-slow detection.

No physical-thermal claims, no MoE-expert modelling, no real-telemetry integration, and no detector in this phase.

## 2. Verified current architecture and limitations

### 2.1 Version and end-to-end pipeline (VERIFIED)

- Upstream: `casys-kaist/LLMServingSim` 2.0 (ISPASS 2026), `README.md:24,42-44`; local `fail_slow` extension HEAD `e691da5`. `CHANGELOG.md:6-21` (11x speedup, 58-scenario `serving/validate.sh`, RTX 4090 profiles).
- Request path: `serving/__main__.py:297,749ff` handshake loop → `router.Router.load_requests():108` parses JSONL → `route_arrived_requests():194` (prefill pool only) → `scheduler.Scheduler.schedule():90` → `trace_generator.generate_trace:1606-1715` → `graph_generator.generate_graph:138-184` (in-process `LLMConverter`) → `controller.Controller.write_flush:47` / `read_wait:22` / `parse_output:54-71` → `sys/id/current=cycle`, `FREQ=1GHz` so cycles == ns → `scheduler.add_done():353` stamps `queuing_delay/TTFT/ITL/latency/TPOT` (`request.py:88-106`).
- Scheduler = vLLM-V1 continuous batching: `running[]` first + tail-preempt only, FCFS `waiting[]`, budgets `max_num_seqs / max_num_batched_tokens / pp_size`, `num_new > 1` = prefill chunk else decode (`scheduler.py:18-31,133-236,272-338`). `Batch` defined `request.py:130`.
- P/D disaggregation: `config_builder.py:579-597` (`pd_type`, prefill `effective_npus = num_npus * 2` shadow receivers); `router.py:17-20`, `scheduler.add_decode:472`, `__main__.py:858-903` handoff. TTFT includes transfer; hold-delays move first-ITL only.
- Parallelism: `_resolve_parallelism:55-136` (`num_npus = tp*pp`, `ep = tp if MoE`), `_resolve_dp_groups:139-227`, topology `[tp(,pp),dp]` → `network.yml`, static for the whole run.
- Timing: compute = profile CSV lookup (`trace_generator._load_perf_db:351-395`: dense 1-D, attention 4-D + skew blend, MoE `(tokens, activated)` per-EP-rank); communication = trace `comm_size` → Chakra `.et` → ASTRA-Sim analytical network (`llm_converter.py`). `Total clocks (ns) = current` (`__main__.py:1457`). Deterministic: identical seeds give identical clocks (`validate.sh:210-250`).
- KV transfer: `Batch.pd_kv_send_tokens` (`request.py:119-136`, `scheduler.py:293-316`) → `qkv_proj.comm_size` (`trace_generator:1047-1059,1266-1280`) → SEND/RECV inside the prefill's own NPU range (`llm_converter.convert_prefill:708-987`, sender `npu_id + num_npus:755`). **Per-pair prefill-i → decode-j targeting is NOT possible** (`docs/fault_injection/ARCHITECTURE_FINDINGS.md:c`).

### 2.2 Area-by-area audit (reuse / extend / replace)

| Area | Key files | VERIFIED state | Tested? | Limitation for day-sim | Verdict |
|---|---|---|---|---|---|
| Arrival / workload | `experiments/fault_detection/workloads.py:27-54`, `workloads/generators/sharegpt.py:404-416`, `router.py:108-231` | Stationary `poisson/bursty/sustained/pulse` + seeded lengths; router accepts arbitrary `arrival_time_ns` | Indirect via paired runs | No `rate(t)` / noon peak | **Reuse + extend** (new rate(t) wrapper, same JSONL) |
| Queues / admission / scheduling | `scheduler.py`, `request.py:20-43,130-165`, `router.py:44-102` | FCFS, `reserve_full_isl`, tail-preempt; `queuing_delay/TTFT/TPOT/ITL` fields | `validate.sh` 58 scenarios | No priority/SLO admission | **Reuse** |
| Batching / prefill / decode | `scheduler._build_batch:272-338`, `request.Batch` | Continuous mixed batching, chunk classification | Validate green | None blocking | **Reuse** |
| GPU / worker / instance | `config_builder:549-606,688-713`, `__main__:598-627,754-830,936-946` | NPU→instance many-to-one; one hw/model/tp per instance; SPMD lockstep completion | Validate green | No per-NPU state or timing; heterogeneous ranks impossible | **Extend** (state table + rank hook, see §4) |
| KV handoff | Above + `held_transfers:438,771-779,879-891` | Bandwidth-size inflation, hold-delay, stall-skip | Fault tests + measured baselines | Per-pair impossible; latency term output-only | **Reuse** |
| Compute / comm timing | `trace_generator` lookups/emitters, `memory_model`, `graph_generator`, `llm_converter` | One latency per layer replicated to all ranks; MoE diverges only per-EP-rank counts | Bench within ~2.5% | No dynamic per-rank multiplier | **Extend** schedule-time hook |
| Fault / slowdown | `fault_injection.py:31-302`, `__main__:1158-1191,922-934,1381-1415` | 6 modes, `severity_weight`, batch-granularity future-only; DP hook absent (`__main__:741-742` rejects DP+faults) | `test_fault_injection.py` (18), 4 recorded baselines | In-flight immutable; link latency static | **Reuse + extend** |
| Telemetry / output | `__main__:1555-1594`, `manifest.py`, `ml/features.py`, `ml/run.py` | batch/request/ITL CSVs + audit + manifest; 18 window features; leak guards (`FORBIDDEN_COLUMNS`, `check_observable_clean`) | `test_fault_ml.py`, audit checks | Per-instance only (`sys` dropped in windows); no util/load/throttle; throughput = batch-count; `request_stage_stats` unused | **Reuse + extend** minimal additive columns |
| Sim time / termination | `__main__:694-714,749-1418`, `_pass_response:65-111` | End-clock `current`, `pass/pass<tick>` wake-ups, held/fault-aware done | Wake-up tests | Day-scale idle-hop cost untested | **Reuse**; perf-check in M2 |
| Tests / experiments | `tests/`, `validate.sh`, `generate.py`, `matrix.yaml`, `docs/fault_injection/RUNLOG.md`, `PROGRESS_REPORT.md` | Paired healthy/faulty, `pair_id`, severity ladder 1.1/1.25/1.5/2.0, smoke 40 windows | 58/58 + smoke green | Day-scale matrix absent | **Reuse + extend** |

Do not infer per-GPU realism from `num_npus`/`tp_size` labels: the converter emits the **same `comp_time`** to every rank (`convert_common:407-409,478-481`, `convert_prefill:739-744,816-819`); power scales uniformly (`power_model.py:73-78`); only MoE `local_tokens` diverges per EP rank (`_emit_moe_block:1105-1195`) with `max` used for power. `fault_injection.apply_to_trace_rows` signature `(faults, rows, role, instance_id, now_s)` cannot address a rank, and non-11-field `EXPERT/PIM` markers are skipped.

## 3. Proposed architecture and data flow (PROPOSED)

```text
Day workload generator (rate(t), seed)
  → workload.jsonl (arrival_time_ns, input/output_toks)   [unchanged schema]
  → Router (LOAD/RR, arrival-sorted) → P-pool queues → P-Schedulers (FCFS)
  → Trace (profile lookup) x GPU-state multiplier (compute-only, schedule-time)
  → ASTRA-Sim → add_done → KV handoff → D-pool queues → D-Schedulers
  → Completions (latency accounting)
  → Telemetry (batch/request/GPU-state windows)  ||  Ground truth (throttle audit + manifest, eval-only)
  → Validation ablations (healthy-peak vs throttled) → future detector data
```

Everything left of `x` is reused; the only new runtime piece is a **per-GPU load/throttle table** consulted at the existing fault-hook point (`__main__:1171-1179`). No scheduler or network-backend replacement.

## 4. GPU topology design (PROPOSED)

- **Logical config (example, not hardcoded):** `prefill_gpus: 4, decode_gpus: 4`, each with `{compute_capacity: 1.0, max_batch_tokens, max_seqs, link_bw/latency (static, from cluster JSON)}`. Pool sizes and all values configurable.
- **Mapping onto existing concepts (smallest defensible):**
  - Operating point for M1: **1 GPU = 1 instance, `tp_size=1, pp_size=1`** (scale the proven `pd_1p1d_small.json` shape to 4P4D). Then `instance_id == GPU id`, and existing `fault_matches(role, instance_id)`, `inst2npu/npu2inst`, telemetry `instance_id`, and `LOAD` routing work unchanged. Document the abstraction gap: an "instance" is the GPU proxy.
  - Extension point for M4: **per-NPU state table** `gpu_state[gpu_id] = {L, throttled, mult}` plus an optional `npu_ids/rank` target in the fault schema; thread rank through `generate_trace → TraceCtx → _emit_layer` (or post-scale per-rank row sets) with a rank-keyed graph cache; teach `LLMConverter.convert_{common,prefill}` per-rank `comp_time`; relax `add_done` SPMD completion. Individual GPU/rank degradation is a design target only — **not built in the first implementation**.
- **Routing/handoff:** keep `Router` P-pool → `transfer_prefill_request` → D-pool; the per-pair transfer limit carries over and is documented.
- **Per-GPU state/telemetry:** new table plus a `sys`-preserving aggregation (see §8). Compute degradation is kept strictly separate from network degradation.

## 5. Daily workload model (PROPOSED)

- **Recommended initial approach:** non-homogeneous Poisson (NHPP) arrivals with a piecewise/configurable `rate(t)`: base + morning ramp + **configurable noon peak (example: Gaussian/sinusoid centred at 12:00 with height/width parameters)** + afternoon decay + evening low, plus seeded jitter and optional short bursts (reuse `bursty`/`pulse` logic). The generator emits the identical JSONL schema (`arrival_time_ns` via thinning of `rate(t)`); prompt/output length distributions reuse `workloads.py:49-54` uniforms first, ShareGPT lengths later.
- **Alternative considered:** trace-driven replay of real arrival timestamps — higher fidelity but needs a suitable public trace plus tokenizer alignment; deferred to M5.
- **Stated limitations:** NHPP assumes independent arrivals; no session correlation, no agentic chains, and no user-retry feedback in M1.
- **Sim-time handling:** no engine change. `current` advances on ASTRA-Sim reports plus `pass`-based idle hops, which already skip sparse night periods; verify day-scale wall-clock cost in M2 (sequential runs, `nohup` + poll per repo convention).
- **End-of-day queued requests:** configurable policy; default **drain with grace period** (finish admitted plus queued work up to `+grace_period`, record `undrained/dropped` counts separately). The alternative truncate-and-mark must be explicit in every report.

## 6. Queueing and timing model (PROPOSED)

Unchanged chain, now time-dependent: arrivals (§5) → per-pool queues (`waiting[]`, FCFS, `reserve_full_isl`, tail-preempt) → prefill execution (profile `comp_time × mult`) → KV SEND/RECV (byte counts unchanged by throttling) → decode queues → decode execution → `add_latency` accounting.

- Demand peaks become visible as `n_waiting/n_running` growth, `queuing_delay`, `TTFT`, `TPOT/ITL`, throughput, and tail latency. Already supported: per-request TTFT/latency/queuing/TPOT/ITL plus batch `duration_ns`. **Needs new instrumentation:** token-throughput series, queue time-series / preempt counts, per-GPU busy fraction (see §8).
- Throttling acts on **queue drain rate**: longer `duration_ns` slows `schedule` turnover, so queues grow whenever arrivals exceed degraded service. Prefill vs decode bottlenecks differ end-to-end (prefill hits TTFT plus transfer occupancy; decode hits ITL/TPOT with backpressure) — exercised as an M5 ablation.

## 7. Exponential-memory degradation model (PROPOSED, synthetic — no hardware claims)

- **Candidate to evaluate (not mandated):** per-GPU `L_i(t+dt) = L_i(t)·exp(−dt/τ) + α·u_i(t)·(1−exp(−dt/τ))`, with dimensionless `L`, normalised busy fraction (or calibrated power proxy) `u ∈ [0,1]`, decay `τ`, and steady-state scale `α` (default `α=1` so `L→u` at steady state).
- **Alternatives:** fixed-window EWMA of busy fraction (simplest; recommended M4 starting point if dt-integration proves fiddly); token-throughput-normalised load; two-state thermal-RC equivalent (rejected for M4 — implies physical parameters we will not claim).
- **Minimal initial formulation:** update `L` per telemetry window from busy fraction; throttle when `L ≥ θ_on`, release when `L ≤ θ_off` (hysteresis with `θ_off < θ_on` prevents flapping); effective compute multiplier `m = 1/(1+γ·f(L))` or stepped `m ∈ {1.0, 0.8, 0.67}`, capped by an **`m_min` safeguard (e.g. 0.5, i.e. max 2x slowdown)**; state persists across fluctuations via `L` decay rather than reset.
- **Healthy vs degraded separation:** healthy queueing correlates with `rate(t)` and `u` while `L<θ`, `m=1`; degraded adds `m<1` with `L≥θ` history. Ablations (§9) include high-load-no-throttle (`θ=∞`) and throttle-under-moderate-load (forced/low-θ) cases.
- **Four-way distinction (must appear in all docs):** (1) synthetic load accumulation, (2) synthetic throttling, (3) physical thermal simulation (**not built** — would need power→heat→ambient→temperature), (4) real hardware calibration (**not built**). The first implementation targets a reproducible, explainable synthetic mechanism only.

Dynamic execution-time policy (PROPOSED): throttling applies to **newly scheduled batches only** (re-evaluated at the existing schedule-time hook, matching the proven `severity_weight` semantics). **Batches already executing are never rescaled** (no mid-run ASTRA-Sim edit API). Compute rows scale; `comm_size`/collectives never scale with throttling. Queue drain, P/D imbalance, and KV-handoff backpressure follow causally from longer compute durations.

## 8. Telemetry schema (PROPOSED, minimal)

Reuse batch/request/ITL CSVs plus the audit/manifest separation (`FORBIDDEN_COLUMNS`, `check_observable_clean`, `check_audit_consistency`). Add compatibly (same pattern as the `com_cycle` precedent; original `--output` CSV untouched):

- Per-window per-GPU (new `gpu_telemetry.csv` or `sys`-preserving extension): `run_id, window_end_s, gpu_id (pool, instance_id), busy_frac, L_state, throttled (0/1), throughput_mult, batch_count, token_throughput, queue_depth_mean/max, ttft_p50/p99, tpot_p50/p99, duration_mean/p99, com_cycle_mean`.
- Request level: keep existing columns; wire the currently unused `request_stage_stats` (TTFT/TPOT/handoff/ITL) into the window vector.
- Ground truth (evaluation-only, never detector input): `throttle_audit.jsonl` (`gpu_id, window, L, u, throttled, mult`) plus manifest labels. The detector path sees operational telemetry only.

## 9. Validation and ablation plan (PROPOSED; invariants, no fabricated numbers)

1. **Healthy steady load:** bounded queues, stable `dur_mean` residuals, zero throttle transitions.
2. **Noon demand peak:** queue/TTFT rise with `rate(t)` while `m=1`; drains post-peak.
3. **Load accumulation:** sustained `u≈1` raises `L` monotonically toward `α·u`.
4. **Recovery:** low-`u` windows decay `L` ≈ `exp(−dt/τ)`.
5. **Throttle onset:** crossing `θ_on` sets `m<1` and inflates `duration_ns` for that GPU's batches only.
6. **Queueing feedback:** throttled bottleneck grows queues vs the unthrottled twin on identical arrivals.
7. **GPU isolation:** throttling GPU k leaves other GPUs' `(L, m)` untouched (shared-queue indirect effects allowed and logged).
8. **Prefill/decode imbalance:** prefill-throttle moves TTFT primarily; decode-throttle moves ITL/TPOT primarily.
9. **Reproducibility:** identical config + seed → identical `Total clocks (ns)` and telemetry hashes.
10. **Backward compatibility:** `validate.sh` 58/58 plus existing smoke and unit tests green with the feature off (disabled path bit-identical).
- Ablations: high-load `θ=∞` (no-throttle) vs throttled twins on shared seeds; forced throttle under moderate load; `rate(t)`-matched healthy vs degraded pairs. Any detection-style scoring is characterised as future-use dataset quality, not a detector claim.

## 10. Milestones and acceptance criteria (PROPOSED)

- **M0 — Baseline (XS).** Freeze SHA; `validate.sh --clocks-only`, unit tests, and a smoke rerun bit-identical. No files change except RUNLOG notes. Accept: 58/58 green, fault baselines match.
- **M1 — Topology (S).** 4P4D (example) 1-GPU=1-instance configs plus routing check; no rank divergence yet. Likely files: `configs/cluster/*`, `config_builder.py` validation only, docs. Accept: all pools schedule, per-instance telemetry separates P/D, determinism holds. Risk: config combinatorial errors.
- **M2 — Workload (S).** NHPP day generator (seeded, configurable noon peak) + drain policy + wall-clock check. Likely files: new `experiments/.../day_workload.py`, `matrix.yaml`, docs. Accept: `rate(t)` reproduces from seed; sparse-night idle hops do not stall; tail policy logged. Risk: day-scale ASTRA-Sim cost — start compressed-day, full-day spot check.
- **M3 — Queueing and telemetry (S–M).** Token-throughput / queue-series / busy-fraction columns plus `request_stage_stats` wiring; leak guards extended. Likely files: `serving/__main__.py` (telemetry only), `ml/features.py`, tests. Accept: peak→queue→TTFT/TPOT/tail visible; guards pass; disabled path identical. Risk: schema churn — additive-only columns.
- **M4 — Stateful degradation (M).** `gpu_state` table + EWMA/exponential `L` + hysteresis throttle; compute-only schedule-time multiplier (future batches only; in-flight immutable documented). Likely files: new `serving/core/gpu_state.py`, `fault_injection.py` extension, `__main__` hook, audit schema. Accept: §9 tests 3–7 pass; `m_min` respected; per-GPU isolation holds. Risk: ASTRA-Sim in-flight limits; DP stays excluded.
- **M5 — Integrated validation (M).** Healthy / peak / degraded day comparisons plus P-vs-D ablations and a labelled dataset spec for future detector work. No detector built. Accept: all §9 ablations reported as normal-vs-slow tables with honest neutrals.

Deferred unless the audit proves essential: physical thermal modelling, real GPU telemetry, MoE-expert simulation, sophisticated detectors.

## 11. Risks, assumptions, and unresolved design decisions (PROPOSED)

- Day-long ASTRA-Sim wall-clock cost is unknown — compressed-day default with a full-day acceptance spot-check.
- `u(t)` definition (busy fraction vs power proxy) — default busy fraction; power proxy optional later.
- `τ, α, θ_on/θ_off, γ, m_min, window dt` defaults need an M4 sensitivity sweep; no values are fixed here beyond the `m_min=0.5` safeguard example.
- In-flight batches are unaffected by throttling (engine limit) — accepted, with onset error bounded by one iteration plus window.
- Per-rank divergence is deferred — M1–M3 proceed per-instance; the rank hook lands in M4 only if needed.
- DP + degradation stays excluded (existing deadlock guard carries over).

## 12. Decisions requiring approval before implementation

1. Approve 1-GPU=1-instance (`tp=1`) as the M1 operating point vs `tp>1` pooled GPUs.
2. Approve the example 4P+4D counts (or set alternate P/D counts) and the compressed-day vs full-24h default.
3. Approve the NHPP-sinusoidal noon-peak generator as the M2 default (vs trace replay).
4. Approve the drain-with-grace vs truncate end-of-day policy.
5. Approve busy-fraction `u`, hysteresis thresholds, and the `m_min=0.5` safeguard example.
6. Approve compute-only throttling (network untouched) with future-batches-only semantics.
7. Approve the additive-only telemetry columns plus the audit/manifest separation as specified.
8. Approve the M0→M5 order and deferral of physical-thermal / MoE-expert / detector work.
9. Acknowledge this file as the planning baseline before any M0–M5 implementation begins.
