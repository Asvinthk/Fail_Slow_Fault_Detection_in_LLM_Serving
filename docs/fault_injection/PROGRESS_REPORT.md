# Two-Month Project: Progress Report
## Fail-slow fault detection for PD-disaggregated LLM serving (simulation-based)

Status legend: DONE = completed, DOING = ongoing, TODO = pending.
This is a prototype for learning, not a production detector.
Synthetic faults cannot prove real-hardware detection.

---

## A. Proposed Approach

### Problem in one paragraph

Production stacks (vLLM, SGLang, Dynamo, Mooncake) split LLM serving
into a prefill pool P, a decode pool D, and transfer links L for KV
cache. A request path is fixed: queue -> prefill at pi -> KV transfer
over lij -> decode at dj. A fail-slow fault keeps working and emits
no error; its only symptom is higher latency. Benign causes (long
prompts, decode-batch mix, MoE routing skew) also raise latency. So
the detector must, per time window W, using only telemetry the
server already emits:

  f(per-hop latencies, request covariates, HW counters)
    -> (alarm in {0,1}, culprit in P u D u L u {workload})

- M1 detect an SLO-relevant slowdown,
- M2 say fault vs workload skew,
- M3 localize to a worker or link.

Metrics: detection delay, top-1/top-3 localization, false-positive
rate under skew, negligible serving overhead.

### Methodology (why simulation first)

Real fail-slow faults are rare, unlabeled, and hard to reproduce.
LLMServingSim 2.0 gives deterministic, profile-driven iteration
latencies (profiler CSVs -> trace -> Chakra graph -> ASTRA-Sim
cycles -> scheduler). We inject controlled fail-slow faults there,
collect observable telemetry plus separate ground truth, then train
transparent baselines first (thresholds, residuals) before any
heavy model. This separates "can we see it?" from "can we learn it?".

Related work surveyed (arXiv ids from the brief): thermal/ECC/
interconnect fail-slow studies, RDMA-to-TCP fallback, MoE skew and
varlen-attention imbalance, PD-disaggregation schedulers. Full
citations live in the final report; the method follows the standard
recipe: healthy-only residual model conditioned on batch shape,
peer-relative features, abstaining localization head.

### Detector sketch (transparent baselines first)

1. Fixed latency thresholds calibrated on one healthy workload.
2. Residual-only detector: healthy-only regression of iteration
   duration vs (total_len, num_prefill, n_decode, kv_len),
   exposed-comm vs pd_kv_send_tokens, ITL vs (n_decode, kv);
   alarm on residual quantile.
3. Workload-aware classifier (logistic regression / GBM) only if
   (2) underperforms on healthy-heavy vs faulty-light pairs.

Features never touch manifest/audit columns (unit-tested).
Splits group by pair_id/workload_id; held-out workload family and
cluster config; time-aware split for detection delay.

## B. System Design

### Architecture (ASCII)

```
Workload JSONL ---> Router ---> Prefill pool P ---> KV link L ---> Decode pool D
                        |              |                |                |
                        v              v                v                v
                   Scheduler --> Trace (comp/comm) --> ASTRA-Sim --> add_done
                                      ^  (FAULT HOOK: scale comp_time /
                                      |   comm_size for target instance)
                                      |
                         fault_injection.py (profiles, targeting, audit)
                                      |
        Observable: batch/request telemetry CSVs   Ground truth: audit jsonl + manifest
                                      |                         |
                                      v                         v
                              Feature builder ---> Baselines ---> Report (P/R/F1, delay, confusion)
```

### Components

| Component | File | Function |
|---|---|---|
| Fault profiles | `serving/core/fault_injection.py: severity_weight` | step/intermittent/ramp w(t) |
| Validation | same file: `validate_fault_config` | reject unknown keys at startup |
| Compute faults | `apply_to_trace_rows` | scale `comp_time` for target role+instance |
| Transfer BW | same | inflate `comm_size` on `qkv_proj` by 1/fraction |
| Handoff delay | `serving/__main__.py` held queue | hold finished prefill before `add_decode` |
| Stall | router skip + schedule guard | no batches in window, requests wait, resume after |
| Audit | `injection_audit.jsonl` | per-batch nominal vs applied, never in telemetry |
| Telemetry | `--telemetry-output` CSVs | batch/request observable stats |
| Workloads | `experiments/fault_detection/workloads.py` | seeded Poisson/bursty JSONL |
| ML baseline | `experiments/fault_detection/ml/` (TODO) | residual + peer-relative features |

### Workflow

1. Generate paired workloads (healthy/faulty share arrivals).
2. Run healthy then faulty with identical seeds; write manifest.
3. Build window features per (run, window, instance, role).
4. Train on healthy only (residuals), classify, localize with
   abstention (`healthy / prefill / decode / kv-transfer-network /
   worker-unavailable / unknown`), plus separate mechanism-class task.
5. Report: PR/F1, FPR on healthy families, localization accuracy +
   macro-F1 + confusion, delay in sim seconds, abstention rate,
   inference time per window.

### Inputs / outputs

Input: cluster JSON (1P1D, 2P2D), workload JSONL
(input_toks, output_toks, arrival_time_ns), fault JSON, seed.
Output: original request CSV (unchanged), telemetry CSVs, audit
jsonl, manifest.jsonl, ML report (CSV + markdown tables).

## C. Technical Feasibility

Tools: Python 3, LLMServingSim 2.0 + ASTRA-Sim analytical backend,
pytest, numpy/scikit-learn (ML stage), Docker sim container
(`scripts/docker-sim.sh`, `scripts/compile.sh`).
Infra: sim container for real runs; this host (Windows, no binary)
runs unit tests only. Data: committed `workloads/example_trace.jsonl`
plus generated Poisson/bursty traces inside profiled sweep bounds
(`meta.yaml: engine_effective`).

Feasible: compute/transfer/handoff/stall hooks are all in the
frontend (verified in source). Not feasible without ASTRA-Sim
changes: mid-run link-latency change, per-pair transfer targeting
(receiver lives in the prefill's own NPU range, Stage 1c), direct
transfer-time separation (proxy only), real crash with loss.

Challenges: batch-granularity onset (<=1 iteration error);
queueing amplification vs direct effect; workload-vs-fault
confusion at low severity (report negatives honestly);
DP configs excluded from prototype (clear error).

Assumptions: profile latencies valid in-bounds; arrivals/lengths
paired across healthy/faulty; detector uses only observable
telemetry; no cluster-wide probing or traffic pausing.

## D. Implementation Plan

| # | Task | Status | Outcome |
|---|---|---|---|
| 1 | Inspect + baseline (Stage 1) | DONE (code reading + findings doc); DOING (binary runs need container) | ARCHITECTURE_FINDINGS.md, RUNLOG.md |
| 2 | Fault subsystem (Stage 2) | DONE (code+unit tests); DOING (validate.sh + PD determinism in container) | `fault_injection.py`, 7 pytest pass, 4 example faults |
| 3 | Telemetry (Stage 3) | DOING (batch CSV stub; request CSV + rolling windows TODO) | original CSV untouched, `com_cycle` added compatibly |
| 4 | Workloads/matrix (Stage 4) | DOING (generator + 1P1D/2P2D configs DONE; full matrix + manifest loader TODO) | smoke-ready scaffolding |
| 5 | Features + ML (Stage 5) | TODO | transparent baselines, one-command report |
| 6 | Docs + verification (Stage 6) | DOING (5 docs DONE; full validate + smoke + ML report TODO in container) | this report |

Next steps (in order): run PD baseline twice + `validate.sh
--clocks-only` in container; add fault scenarios to validate.sh;
finish request telemetry; implement manifest + loader guard;
generate smoke matrix; run ML baselines; record all outputs in
RUNLOG.md. One logical change per commit, short imperative messages.
