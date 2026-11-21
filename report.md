# Fail-Slow Detection in PD-Disaggregated LLM Serving

## Project summary

This project studies how to identify slowdowns in large language model (LLM) inference systems that separate prompt processing (prefill) from token generation (decode). Requests also transfer key/value (KV) cache data between those stages. A fail-slow fault leaves a component operational but increases latency, which can be difficult to distinguish from normal variation in prompt length, output length, batching, and queue load.

The prototype extends the LLMServingSim serving simulator with synthetic fault injection and request, queue, batch, and audit telemetry. The simulator uses profiled operation timings and analytical system simulation. It does not run Qwen3-32B weights during this experiment and does not measure physical GPU temperature or real network behavior. Results are simulator evidence only.

## Models implemented

The selected prototype contains two models.

### 1. Healthy-versus-faulty classifier

A logistic regression model distinguishes requests from a healthy replay from requests in a combined synthetic-fault run. The paired Qwen3-32B experiment uses 300 requests in each condition, with matching request IDs 0–299. Its classifier uses workload and queue context together with observed prefill, KV handoff/transfer, and decode timing telemetry. It does not use labels, fault IDs, audit/ledger data, or injected-time-delta fields as features.

Preprocessing (median imputation and standard scaling) is fitted on the training partition. A seed-23, 75/25 holdout groups matched request IDs together, preventing the healthy and faulty versions of one request from crossing partitions.

| Held-out metric | Result |
|---|---:|
| Accuracy | 100.00% |
| Balanced accuracy | 100.00% |
| Faulty precision | 100.00% |
| Faulty recall | 100.00% |
| Faulty F1 | 100.00% |
| ROC AUC | 1.0000 |

Confusion matrix (true rows; predicted columns; healthy, faulty):

```text
[[75,  0],
 [ 0, 75]]
```

This is retrospective request classification because it uses observed stage/completion timing. It does not establish early warning, detection delay, SLO-threshold accuracy, or generalization to a different workload or topology.

### 2. Healthy-only stage-time regressors

Linear regression models estimate prefill service time, KV handoff interval, and aggregate decode time. They are trained only on the 300-request healthy replay. Rows are ordered by arrival; the first 225 are used for training and the final 75 for testing. Each regressor uses workload and pre-service queue features; observed target timings are excluded from its input. Preprocessing is fitted on training rows only.

| Stage | Target | MAE (ms) | RMSE (ms) | R² | Mean baseline MAE (ms) |
|---|---|---:|---:|---:|---:|
| Prefill | First scheduled to prefill completion | 1.661 | 1.979 | 0.9909 | 18.592 |
| KV transfer/handoff | Decode admission minus prefill completion | 0.000 | 0.000 | 1.0000 | 0.000 |
| Decode | Mean inter-token latency × token interval count | 12.113 | 18.738 | 0.9999 | 1,756.356 |

The healthy run records a zero transfer interval for every request, so the perfect KV result is a constant-target case, not evidence of learned transfer timing. The high decode score is specific to this workload and needs evaluation on other workloads.

## Data and interpretation

The two runs have 300 unique, aligned request IDs and matching arrival times, prompt lengths, and requested output lengths. The faulty run combines KV-transfer bandwidth reduction, collective-network slowdown, KV handoff delay, and synthetic thermal-like throttling. Since these perturbations co-occur, this report does not claim component-level fault attribution.

The fault ledger records compute deltas and communication-volume changes. It does not provide exact per-request healthy communication timing to subtract from the faulty run. The healthy replay timings above are used as the measured healthy regression targets.

## Current scope and next work

The current implementation reports binary healthy-versus-faulty classification and healthy-only stage-time regression. Component localization is outside this selected prototype. The models need multi-seed and multi-workload evaluation. The regressors are not connected to an alarm threshold, and no online detection delay or serving overhead has been measured.

All injected faults are synthetic. These experiments validate the telemetry and analysis workflow at small scale; they do not demonstrate detection of real hardware faults.

## Reproducibility files

- Training code: `research/fault_detection/train_qwen32_binary_and_stage_regressors.py`
- Detailed results: `research/docs/fault_injection/QWEN32_SINGLE_300_TWO_MODEL_RESULTS.md`
- Presentation-generation instructions: `research/docs/fault_injection/CLAUDE_FINAL_SUBMISSION_DECK_BRIEF.md`
- Exact executed command/output log: `research/docs/fault_injection/RUNLOG.md`
- Generated model reports and binaries: ignored `research/out/qwen32_single_300_uniform_training/model_results/`

Re-run from the repository root after placing the two existing run folders under `research/out/qwen32_single_300_uniform_training/`:

```powershell
python research/fault_detection/train_qwen32_binary_and_stage_regressors.py --data-root research/out/qwen32_single_300_uniform_training --out research/out/qwen32_single_300_uniform_training/model_results --seed 23
```
