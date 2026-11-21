# Claude deck creation brief — final project submission

Create an editable university research presentation and a PDF preview for **Fail-Slow Detection in PD-Disaggregated LLM Serving**. Use the attached earlier presentation as a visual reference for its dark technical theme, diagrams, and restrained accent colors. Improve legibility and use clean, readable type. Do not copy old results when this brief gives updated evidence. Do not invent facts. Target 18 slides unless the source presentation and presentation constraints make that impossible; do not force content into tiny text. No video is requested: the system behavior is a simulator and data-analysis workflow, so the demonstration is a reproducible live walkthrough.

## Audience and story

Audience: university project evaluator who may ask implementation and methodology questions. Start with the serving problem, explain what the simulator computes, show the actual implemented telemetry/training workflow, report results with appropriate caveats, then separate completed work from limitations and next steps. Plain language first; define PD, KV, ITL, and fail-slow before using abbreviations heavily.

## Core problem

In prefill/decode (PD) disaggregation, requests queue, run prefill, transfer KV-cache state, then decode tokens. A fail-slow component remains operational but becomes slower. Workload variation can look similar, so a useful system should detect slowdown (M1), distinguish component degradation from workload skew (M2), and ideally localize a culprit (M3). This project's selected prototype implements only healthy-versus-faulty classification and healthy-only timing regression; component localization is outside its scope. It uses synthetic faults in a serving simulator and does not validate detection on real hardware.

## Explain the simulator accurately

Show this data path clearly:

`vLLM profile timing tables + model/config + request arrivals → serving scheduler and batching → shape-dependent compute/communication trace → Chakra conversion → ASTRA-Sim analytical execution → simulated request and system timing`

The laptop/server does not execute Qwen3-32B weights in the studied workflow. ASTRA-Sim models configured compute/network work; it is not a physical GPU or live-network measurement. Request times are simulator estimates under the profile and topology assumptions. Do not claim physical temperature, ECC, PCIe/NVLink behavior, or measured cluster timing.

Separate upstream LLMServingSim scheduling/profile/trace/simulation behavior from this project's synthetic injection hooks, telemetry exports, audit/ledger capture, paired analysis, and ML experiments. Ground-truth labels and audits are evaluation-only and excluded from model features.

## Selected two-model workflow

Show the two independent models:

1. **Binary healthy-versus-faulty logistic regression:** 300 healthy replay rows + 300 combined-fault rows, same 300 request IDs. Use workload/queue context plus directly observed prefill, KV handoff/transfer, and decode timing telemetry. Exclude label, fault IDs, audits, ledger, manifest, injected added-time/delta fields, request IDs, and absolute timestamps. Median imputation and standard scaling are fitted on the training partition only. Reproducible seed 23; 75/25 split grouped by request ID so matched rows stay together. Held-out metrics: accuracy 100.00%, balanced accuracy 100.00%, faulty precision/recall/F1 100.00%, ROC AUC 1.0000; confusion matrix (true rows, predicted columns: healthy/faulty) `[[75, 0], [0, 75]]`. This uses observed request timing, so call it retrospective request classification, not early warning or proof of unseen-workload generalization.
2. **Healthy-only stage-time regressors:** trained only on healthy rows; chronological split 225 train / 75 test; preprocessing fitted only to training. Use workload and pre-service queue features only; exclude each observed stage target and all completed timing telemetry from that target's predictors. Prefill service = first scheduled to prefill completion. Transfer target = decode admission minus prefill completion. Decode target = mean ITL times ITL count (aggregate proxy). Held-out metrics (MAE/RMSE/R²): prefill 1.661/1.979 ms/0.9909; KV transfer 0/0 ms/1.0000; decode 12.113/18.738 ms/0.9999. The transfer interval is zero for every healthy row; its perfect score is a degenerate constant target, not learned transfer prediction. The decode fit reflects this workload and needs cross-workload validation.

Regressors are a baseline prototype only: no residual threshold or per-request alarm is implemented. Component localization is out of scope for this selected two-model prototype; do not show a third classifier or component localization metrics.

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
9. Two-model architecture and feature/label separation.
10. Paired dataset, request-ID audit, and split strategy.
11. Logistic-regression results with confusion matrix.
12. Stage regression targets and train-only preprocessing.
13. Stage regression metrics and zero-transfer caveat.
14. Counterfactual limits and distinction between measured replay and ledger deltas.
15. End-to-end demonstration walkthrough (load existing telemetry, inspect IDs/features, run trainer, inspect metrics/artifacts; no video).
16. Limitations and complete/pending status.
17. Next experiments and reproducibility evidence.
18. Conclusion and selected sources.

## Demo plan slide

Present a short live terminal/notebook walkthrough using already completed runs: (a) show the two run manifests and 300-row exports, (b) run the reproducible trainer command, (c) display `training_summary.json`, classifier confusion matrix, and stage metrics, (d) show that the selected design contains only the binary classifier and healthy-only regressors, (e) point to run log and report. Do not run new simulations as part of the demo.

## Visual and evidence rules

- Match the earlier deck's dark technical identity, with consistent fonts, high contrast, and simple stage diagrams.
- Use readable tables and one strong evidence chart; do not shrink copy to fit.
- No repeated footer/source strip. Put a small selected-source list on the final slide.
- Show the binary classifier and the separate healthy-only timing regressors; do not include a component localizer in this version.
- No claim that synthetic faults prove real-hardware detection.
- No invented localization scores, detection latency, false-positive rate under workload skew, SLO threshold validation, or serving overhead measurement.
- Mark status: complete (ID audit, binary model using measured timing telemetry, healthy stage regressors), pending (multi-run/workload evaluation, residual alarm validation, deployment evidence).

## Selected official simulator sources

- [LLMServingSim overview](https://llmservingsim.ai/docs/getting-started/overview)
- [Profiler overview](https://llmservingsim.ai/docs/profiler/overview)
- [Profiler output bundle](https://llmservingsim.ai/docs/profiler/output-bundle)
- [Trace generation](https://llmservingsim.ai/docs/simulator/trace-generation)
- [Trace format and Chakra conversion](https://llmservingsim.ai/docs/reference/trace-format)
- [Continuous batching](https://llmservingsim.ai/docs/simulator/scheduling/continuous-batching)
- [ASTRA-Sim repository](https://github.com/astra-sim/astra-sim)

## Project sources

- `QWEN32_SINGLE_300_TWO_MODEL_RESULTS.md` (authoritative for the selected two-model prototype's new 300-row results)
- `RUNLOG.md` (executed commands and exact outputs)
- Existing `QWEN32_PRESENTATION_BRIEF_AND_DEEPSEEK_PROMPT.md`, `QWEN32_PER_FAULT_RESULTS.md`, `QWEN32_SERVICE_TIME_REGRESSION.md`, and `QWEN32_FAULT_SLOWDOWN.md` for earlier study context only; do not mix their 1,000-row results into the new 300-row experiment.
