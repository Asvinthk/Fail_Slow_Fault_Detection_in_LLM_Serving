# Qwen3-32B 300-request three-model prototype

## Status and scope

This report records training on the existing paired Qwen3-32B 2P2D TP2 serving-simulator runs. Each run contains the same 300 requests and the same uniform workload. It is a synthetic simulator study, not evidence of real hardware fault detection.

The faulty folder name says `four_faults`, but its `fault_config.json` contains **three** mechanism-level entries: KV-transfer bandwidth at 70%, collective-network degradation at 1.3x, and a 20 ms KV handoff delay. Thermal throttling is configured separately for GPUs 0, 1, 4, 5, 8, 9, 10, and 11. All run concurrently. No direct prefill-compute or decode-compute fault is configured in this pair.

## Data preparation and ID audit

The existing exporters produced `request_training_data.csv` for both runs. Reconciliation verified 300 unique IDs in each, IDs 0–299, identical ID sets, and matching arrival time, prompt length, and requested output length for every paired request. The classifier dataset therefore contains 600 rows: 300 healthy and 300 combined-fault condition examples.

Model features are restricted to prompt length, requested output length, inter-arrival gap, routing lag, and pre-service queue snapshot counts. Excluded fields include request IDs, instance IDs, absolute timestamps, labels, fault IDs/configuration, audit/ledger/manifest fields, and added-compute or counterfactual fields. Median imputation and standard scaling are fitted inside each model pipeline on the training partition only.

Healthy replay stage targets are directly measured from the healthy request telemetry. We do not subtract communication deltas from faulty-run timing: the ledger provides communication-volume changes, not exact per-request healthy communication timing. The healthy run's transfer interval is zero for every request, so its transfer regressor is a degenerate constant-target baseline.

## Model 1: healthy versus combined-fault condition

Logistic regression uses a reproducible 75/25 grouped holdout (seed 23), grouping the two rows with the same request ID so a request's healthy and faulty pair cannot cross the split. Training has 450 rows (225 per condition); test has 150 rows (75 per condition).

| Metric | Held-out result |
|---|---:|
| Accuracy | 93.33% |
| Balanced accuracy | 93.33% |
| Faulty precision | 100.00% |
| Faulty recall | 86.67% |
| Faulty F1 | 92.86% |
| ROC AUC | 0.9877 |

Confusion matrix (true rows; predicted columns; healthy, faulty):

```text
[[75,  0],
 [10, 65]]
```

This uses admission-time/workload features. It discriminates the two matched run conditions on one workload and one topology. The label is run-condition membership, not independently verified SLO violation. One trace cannot establish generalization, detection delay, or false-positive rate under unseen workload skew.

## Model 2: three-way component localizer

**Not trained.** Although the intended classes are prefill, decode, and KV transfer, these runs cannot supply valid exclusive request-level labels for all three. The configured transfer and network mechanisms and throttle co-occur. Fault audits and ledgers are batch/component-level; they do not assign a single responsible component to each request. `prefill_added_compute_share_ns` and `decode_added_compute_share_ns` are compute ledger allocations, not exclusive ground-truth cause labels. Training on them would invent labels, so no localizer metrics or model artifact are reported. A valid localization experiment needs isolated component-specific runs and an explicit request-to-fault exposure mapping.

The classifier-to-localizer cascade is consequently only partially implemented: the classifier can gate a future localizer, but there is no truthful localizer to invoke for this dataset.

## Model 3: healthy-only stage-time regressors

Linear regressors are trained only on the 300 healthy rows. Requests are sorted by arrival time; the first 225 train and final 75 test. Each pipeline's imputer and standard scaler are fitted on its training partition only. Features are the same pre-service features listed above.

| Stage target | Definition | MAE (ms) | RMSE (ms) | R² | Mean baseline MAE (ms) |
|---|---|---:|---:|---:|---:|
| Prefill | first scheduled to prefill complete | 1.661 | 1.979 | 0.9909 | 18.592 |
| KV transfer/handoff | decode admission minus prefill completion | 0.000 | 0.000 | 1.0000 | 0.000 |
| Decode | sum proxy = mean ITL × ITL count | 12.113 | 18.738 | 0.9999 | 1,756.356 |

The decode result's high R² is associated with this workload's token-length pattern and is not evidence of cross-workload accuracy. The zero-transfer result only reflects the simulator's zero observed transfer interval in this healthy run. These regressors are baseline prototypes: no residual threshold, per-example alarm, or fault inference is implemented here.

## Reproduction

Inputs are staged from the lab server under the ignored path `research/out/qwen32_single_300_uniform_training/`. Run from the repository root:

```powershell
python research/fault_detection/train_qwen32_uniform_three_models.py --data-root research/out/qwen32_single_300_uniform_training --out research/out/qwen32_single_300_uniform_training/model_results --seed 23
```

The command writes `training_summary.json`, classifier/regression reports, and fitted `.joblib` models under the ignored output directory. The Python trainer is committed; the paired raw outputs and fitted models remain generated data and are gitignored.

## Completion status

- **Complete:** data export availability, 300-ID alignment audit, leak-excluded features, binary logistic regression, healthy-only stage regressors, train-only preprocessing, held-out metrics and confusion matrix.
- **Partial:** classifier can gate a localizer in the design, but no localizer was fit because per-request three-class labels are unsupported by the combined-fault data.
- **Pending:** isolated prefill/decode/KV-transfer fault runs with request-level attribution; multi-seed and multi-workload evaluation; residual threshold calibration and online alarm evaluation; final submission deck/report integration and push review.

## Exact command output

See `RUNLOG.md` for the exact host and lab commands and captured output.
