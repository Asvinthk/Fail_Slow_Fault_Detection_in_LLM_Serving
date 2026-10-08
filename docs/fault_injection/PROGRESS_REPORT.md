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

Metrics (brief's targets; currently implemented subset: binary
precision/recall/F1, healthy FPR, stub localisation accuracy with
abstention rate, detection delay vs window floor, ms per window —
top-1/top-3 ranking, macro-F1, confusion matrices, per-family FPR,
and serving-overhead measurement are NOT implemented): detection
delay, top-1/top-3 localization, false-positive rate under skew,
negligible serving overhead.

### Methodology (why simulation first)

Real fail-slow faults are rare, unlabeled, and hard to reproduce.
LLMServingSim 2.0 gives deterministic, profile-driven iteration
latencies (profiler CSVs -> trace -> Chakra graph -> ASTRA-Sim
cycles -> scheduler). We inject controlled fail-slow faults there,
collect observable telemetry plus separate ground truth, then train
transparent baselines first (thresholds, residuals) before any
heavy model. This separates "can we see it?" from "can we learn it?".

LLMServingSim 2.0 is NOT an inference server. It is a cycle-level
simulator: a Python frontend mirrors vLLM-style continuous batching
and scheduling, but it executes no model weights — compute timing
comes from layerwise latency profiles measured with a vLLM-based
profiler, and ASTRA-Sim analytically models communication cycles
from Chakra graphs. Hardware is therefore "profiled" or "measured
on", never "trained on". This project simulates RTX PRO 6000 +
Llama-3.1-8B; Qwen3-32B profiles are present for scale-up, while
Llama-3.1-70B has a config but no profile tables and cannot be
simulated.

The current detector operates on simulator-visible serving telemetry
only (batch durations, queues, stage stamps, TTFT/ITL stats, KV-send
counts, exposed-communication proxy). Physical hardware counters from
the broader problem statement — GPU power/thermal state, SM/GPU
utilisation, NIC/RDMA counters, clocks/throttling — are not exposed
through the simulator detection path.

Related work actually present in this repo (no complete
PagedAttention, Sarathi-Serve, or speculative-decoding citations
exist here — do not cite them as surveyed): SGLang RadixAttention
attribution for prefix caching (`CHANGELOG.md`, `CONTRIBUTORS.md`);
DistServe + Mooncake links for PD disaggregation
(`docs/docs/examples/disaggregated/prefill-decode-split.mdx`);
speculative decoding only as a contributor-topic bullet, not a
method. The method follows the standard recipe: healthy-only
residual model conditioned on batch shape, peer-relative features,
abstaining localization head.

### Detector sketch (transparent baselines first — implemented)

1. Fixed latency thresholds calibrated on one healthy workload
   (`ml/run.py`: p99 of window `dur_mean` on train-healthy windows).
2. Residual-only detector: healthy-only least-squares regression of
   iteration duration vs (total_len, num_prefill, n_decode, kv_len)
   plus exposed-comm vs pd_kv_send_tokens (`ml/features.py`); alarm
   on the residual quantile.
3. Workload-aware classifier: logistic regression on the window
   feature vector. Gradient boosting is NOT implemented (reserved
   for when residuals demonstrably underperform). No MLP of any kind
   is implemented, and none is claimed.
4. Localisation is currently a STUB with abstention, not a learned
   model: confident-anomalous windows are labeled from ground truth
   (`ml/run.py`), so its accuracy figure is not a model result and a
   genuine localisation head is pending work.

Features never touch manifest/audit columns (unit-tested
`FORBIDDEN_COLUMNS` guard). Splits group by pair_id/workload_id
(train/test only; no validation split yet); held-out workload family
and cluster config go to test; detection delay measured as
first-anomalous-window minus fault onset against the window floor.

## B. System Design

### Architecture (ASCII)

```
Workload JSONL ---> Router ---> Prefill pool P ---> KV link L ---> Decode pool D
                        |              |                |                |
                        v              v                v                v
                   Scheduler --> Trace (comp/comm) --> ASTRA-Sim --> add_done
                                     ^  (FAULT HOOK, side path only:
                                     |   scale comp_time / comm_size,
                                     |   handoff holds, stall guards)
                                     |
                        fault_injection.py (profiles, targeting, audit)
                                     |
        Observable: batch/request telemetry CSVs   Ground truth: audit jsonl + manifest
        (ONLY detector input)                      (evaluation only: NOT PROVIDED TO DETECTOR)
                                     |                         |
                                     v                         v
                    Feature Extraction --> Healthy Baseline --> Residual/peer features
                                                               --> Fault Detector --> Fault Localiser
                                                               --> Diagnosis / Abstain
```
(Diagram as an image: `experiments/fault_detection/smoke/architecture.svg`.
Detection runs BEFORE localisation; nothing bypasses Feature
Extraction; no arrow runs from Fault Injection to Healthy Baseline.)

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
4. Score windows with the three baselines; localisation stub with
   abstention (`healthy / prefill / decode / kv-transfer-network /
   worker-unavailable / unknown`). No separate mechanism-classification
   task is implemented.
5. Report: binary precision/recall/F1, FPR on healthy windows,
   stub localisation accuracy, abstention + selective accuracy,
   detection delay in sim seconds, inference ms per window. NOT
   implemented: macro-F1, confusion matrices, top-1/top-3 ranking,
   per-family FPR breakdowns, serving-overhead measurement.

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

COMPLETED (verified against code and artifacts before writing):

- fault injection (5 simulator modes, strict validator, audit rules)
- telemetry (batch/request CSVs, ITL side file, dedupe + id-1 mapping)
- workloads (seeded Poisson/bursty/sustained generator)
- paired experiment framework (`generate.py`, shared workload + pair_id)
- manifest / leak guards (`manifest.py`, unit-tested)
- three detection baselines (fixed threshold, residual threshold,
  logistic regression) with abstaining localisation stub
- deterministic validation (58/58 suite green + 4 recorded fault
  baselines; reruns bit-identical)
- smoke evaluation (3 pairs, 5 runs, 0 failed; 40-window ML report)

ONGOING:

- Qwen3-32B scale-up preparation (config + full profile tables verified
  present; no fault runs yet)

PENDING (not implemented; do not claim):

- Qwen3-32B experiments; full 38-job fault matrix; genuine
  localisation model; top-1/top-3 localisation; macro-F1/confusion
  matrix; per-family FPR; workload-skew evaluation at scale; serving
  overhead measurement; mechanism classification; broader
  generalisation study

| # | Task | Status | Outcome |
|---|---|---|---|
| 1 | Inspect + baseline (Stage 1) | COMPLETED | ARCHITECTURE_FINDINGS.md; healthy 1661784496 ns x3 identical |
| 2 | Fault subsystem (Stage 2) | COMPLETED | `fault_injection.py`; 18 injection tests; drop/hold bug found by measurement and fixed |
| 3 | Telemetry (Stage 3) | COMPLETED | full batch/request schema; `com_cycle` added compatibly; original CSV untouched |
| 4 | Workloads/matrix (Stage 4) | COMPLETED | generator + 1P1D/2P2D configs + matrix + paired runner + manifest loader |
| 5 | Features + ML (Stage 5) | COMPLETED as baselines | 3 detectors + stub; one-command report on 40 smoke windows |
| 6 | Docs + verification (Stage 6) | COMPLETED (this cycle) | validate green; smoke bundle committed; statuses reconciled |

Next steps (in order): Qwen3-32B 1P1D config; healthy re-baseline
x2; same-severity fault set + severity ladder; `pd_qwen_fault_*`
baselines; Qwen smoke + ML report; 8B-vs-32B detectability
comparison. One logical change per commit, short imperative messages.
