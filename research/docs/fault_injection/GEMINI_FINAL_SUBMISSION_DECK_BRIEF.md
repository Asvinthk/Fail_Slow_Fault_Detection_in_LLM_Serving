# Gemini deck creation brief — final project submission

Create an editable university research presentation and a PDF preview for **Fail-Slow Detection in PD-Disaggregated LLM Serving**. Use the attached earlier presentation as a visual reference for its dark technical theme, diagrams, and restrained accent colors. Improve legibility and use clean, readable type. Do not copy old results when this brief gives updated evidence. Do not invent facts. Target 18 slides unless the source presentation and presentation constraints make that impossible; do not force content into tiny text. No video is requested: the system behavior is a simulator and data-analysis workflow, so the demonstration is a reproducible live walkthrough.

## Audience and story

Audience: university project evaluator who may ask implementation and methodology questions. Start with the serving problem, explain what the simulator computes, show the actual implemented telemetry/training workflow, report results with appropriate caveats, then separate completed work from limitations and next steps. Plain language first; define PD, KV, ITL, and fail-slow before using abbreviations heavily.

## Core problem

In prefill/decode (PD) disaggregation, requests queue, run prefill, transfer KV-cache state, then decode tokens. A fail-slow component remains operational but becomes slower. Workload variation can look similar, so a useful system should detect slowdown (M1), distinguish component degradation from workload skew (M2), and localize a culprit in prefill, decode, or transfer (M3). This project's prototype uses synthetic faults in a serving simulator; it does not validate detection on real hardware.

## Explain the simulator accurately

Show this data path clearly:

`vLLM profile timing tables + model/config + request arrivals → serving scheduler and batching → shape-dependent compute/communication trace → Chakra conversion → ASTRA-Sim analytical execution → simulated request and system timing`

The laptop/server does not execute Qwen3-32B weights in the studied workflow. ASTRA-Sim models configured compute/network work; it is not a physical GPU or live-network measurement. Request times are simulator estimates under the profile and topology assumptions. Do not claim physical temperature, ECC, PCIe/NVLink behavior, or measured cluster timing.

Separate upstream LLMServingSim scheduling/profile/trace/simulation behavior from this project's synthetic injection hooks, telemetry exports, audit/ledger capture, paired analysis, and ML experiments. Ground-truth labels and audits are evaluation-only and excluded from model features.

## Updated three-model workflow

Show a cascade:

1. **Binary logistic classifier:** 300 healthy replay rows + 300 combined-fault rows, same 300 request IDs. Features: workload and pre-service queue/routing covariates. Median imputation and standard scaling fitted on training partition. Reproducible seed 23; 75/25 split grouped by request ID to keep matched pairs together. Held-out confusion matrix, true rows/predicted columns (healthy, faulty): `[[75, 0], [10, 65]]`. Accuracy 93.33%, balanced accuracy 93.33%, faulty precision 100.00%, recall 86.67%, F1 92.86%, ROC AUC 0.9877. This is run-condition discrimination on one matched workload, not verified SLO alarm performance, detection delay, or unseen-workload generalization.
2. **Three-way localizer:** intended classes prefill/decode/KV transfer and invoked only after a positive classifier result. **Not trained on this dataset.** The faulty run has simultaneous KV-transfer bandwidth, collective-network, and handoff-delay mechanisms plus thermal throttle; its audits are batch/component-level and do not give valid exclusive per-request labels for all three classes. Do not make up labels or localization scores. Mark this stage as pending isolated component-labeled data.
3. **Healthy-only stage-time regressors:** trained only on healthy rows; chronological split 225 train / 75 test; preprocessing fitted only to training. Prefill service = first scheduled to prefill completion. Transfer target = decode admission minus prefill completion. Decode target = mean ITL times ITL count (aggregate proxy). Results:

| Stage | MAE ms | RMSE ms | R² | Mean baseline MAE ms |
|---|---:|---:|---:|---:|
| Prefill | 1.661 | 1.979 | 0.9909 | 18.592 |
| KV transfer/handoff | 0.000 | 0.000 | 1.0000 | 0.000 |
| Decode | 12.113 | 18.738 | 0.9999 | 1,756.356 |

Explain that the transfer interval is zero for every healthy row; the perfect score is a degenerate constant target, not learned transfer prediction. The high decode score reflects this particular workload's token pattern and needs cross-workload validation. Regressors are a baseline prototype only: no residual threshold or per-request alarm is implemented.

## Dataset and counterfactual honesty

Both runs are Qwen3-32B, 2P2D TP2 with the same uniform 300-request workload. All IDs 0–299 align exactly, and arrival/prompt/requested-output covariates match. The faulty folder label says “four faults,” but the fault configuration lists three mechanism entries (KV transfer bandwidth 0.70, collective network degradation 1.30, KV handoff delay 20 ms) plus a separate thermal throttle configuration. State this clearly and avoid claiming separate prefill/decode compute injection in these runs.

The faulty ledger contains compute deltas and communication-volume changes, not exact per-request healthy communication timing. Healthy replay timings are measured healthy targets; do not say communication timing was exactly subtracted or counterfactually recovered.

## Suggested 18-slide sequence

1. Title and bounded contribution.
2. PD request path and fail-slow problem.
3. M1/M2/M3 objectives and what counts as evidence.
4. Why workload skew creates false alarms.
5. How LLMServingSim creates simulated time (pipeline diagram above).
6. Upstream simulator versus this project extension.
7. System/configuration under study: Qwen3-32B, 2P2D TP2, same topology for the paired runs.
8. Existing fault mechanisms and telemetry (careful actual configuration).
9. Three-model architecture and feature/label separation.
10. Paired dataset, request-ID audit, and split strategy.
11. Logistic-regression results with confusion matrix.
12. Localization stage: why it is not trained; exact additional evidence required.
13. Healthy-only regression targets and train-only preprocessing.
14. Stage regression metrics and zero-transfer caveat.
15. Counterfactual limits and distinction between measured replay and ledger deltas.
16. End-to-end demonstration walkthrough (load existing telemetry, inspect IDs/features, run trainer, inspect metrics/artifacts; no video).
17. Limitations, complete/partial/pending status, next experiments.
18. Conclusion and selected sources.

## Demo plan slide

Present a short live terminal/notebook walkthrough using already completed runs: (a) show the two run manifests and 300-row exports, (b) run the reproducible trainer command, (c) display `training_summary.json`, confusion matrix and stage metrics, (d) show that no localizer model was emitted because labels are invalid, (e) point to run log and report. Do not run new simulations as part of the demo.

## Visual and evidence rules

- Match the earlier deck's dark technical identity, with consistent fonts, high contrast, and simple stage diagrams.
- Use readable tables and one strong evidence chart; do not shrink copy to fit.
- No repeated footer/source strip. Put a small selected-source list on the final slide.
- Keep classifier, localizer, and regressor visually distinct. Show that classifier-positive is the intended localizer gate and the regressor is a separate healthy baseline path.
- No claim that synthetic faults prove real-hardware detection.
- No invented localization scores, detection latency, false-positive rate under workload skew, SLO threshold validation, or serving overhead measurement.
- Mark status: complete (ID audit, binary model, healthy stage regressors), partial (cascade without localizer), pending (isolated labels, multi-run/workload evaluation, residual alarm and deployment evidence).

## Selected official simulator sources

- [LLMServingSim overview](https://llmservingsim.ai/docs/getting-started/overview)
- [Profiler overview](https://llmservingsim.ai/docs/profiler/overview)
- [Profiler output bundle](https://llmservingsim.ai/docs/profiler/output-bundle)
- [Trace generation](https://llmservingsim.ai/docs/simulator/trace-generation)
- [Trace format and Chakra conversion](https://llmservingsim.ai/docs/reference/trace-format)
- [Continuous batching](https://llmservingsim.ai/docs/simulator/scheduling/continuous-batching)
- [ASTRA-Sim repository](https://github.com/astra-sim/astra-sim)

## Project sources

- `QWEN32_SINGLE_300_THREE_MODEL_RESULTS.md` (authoritative for the new 300-row model results)
- `RUNLOG.md` (executed commands and exact outputs)
- Existing `QWEN32_PRESENTATION_BRIEF_AND_DEEPSEEK_PROMPT.md`, `QWEN32_PER_FAULT_RESULTS.md`, `QWEN32_SERVICE_TIME_REGRESSION.md`, and `QWEN32_FAULT_SLOWDOWN.md` for earlier study context only; do not mix their 1,000-row results into the new 300-row experiment.
