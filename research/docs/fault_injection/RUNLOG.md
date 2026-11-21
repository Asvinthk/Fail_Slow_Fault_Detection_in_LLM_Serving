SSH_OK
DIR:/home/es23btech11009/LLMServingSim/LLMServingSim/research/out/qwen32_single_300_uniform_four_faults
total 1012
-rw-rw-r-- 1 es23btech11009 es23btech11009    432 Nov 21 17:52 dataset_manifest.json
-rw-rw-r-- 1 es23btech11009 es23btech11009      8 Nov 21 16:59 driver.pid
-rw-rw-r-- 1 es23btech11009 es23btech11009   2897 Nov 21 17:52 export_uniform_dataset.py
-rw-rw-r-- 1 es23btech11009 es23btech11009    776 Nov 21 16:59 fault_config.json
-rw-rw-r-- 1 es23btech11009 es23btech11009  21685 Nov 21 16:59 input_workload.jsonl
-rw-rw-r-- 1 es23btech11009 es23btech11009  69693 Nov 21 17:52 request_training_data.csv
-rw-rw-r-- 1 es23btech11009 es23btech11009 434804 Nov 21 17:17 simulation.log
-rw-rw-r-- 1 es23btech11009 es23btech11009 467422 Nov 21 17:17 simulation_output.csv
drwxrwxr-x 3 es23btech11009 es23btech11009   4096 Nov 21 17:17 telemetry
-rw-rw-r-- 1 es23btech11009 es23btech11009   1707 Nov 21 16:59 throttle_config.json
DIR:/home/es23btech11009/LLMServingSim/LLMServingSim/research/out/qwen32_single_300_uniform_healthy
total 976
-rw-rw-r-- 1 es23btech11009 es23btech11009      8 Nov 21 17:24 driver.pid
-rw-rw-r-- 1 es23btech11009 es23btech11009    676 Nov 21 17:52 export_healthy.py
-rw-rw-r-- 1 es23btech11009 es23btech11009  21685 Nov 21 17:24 input_workload.jsonl
-rw-rw-r-- 1 es23btech11009 es23btech11009  54469 Nov 21 17:52 request_training_data.csv
-rw-rw-r-- 1 es23btech11009 es23btech11009 432470 Nov 21 17:42 simulation.log
-rw-rw-r-- 1 es23btech11009 es23btech11009 466643 Nov 21 17:42 simulation_output.csv
drwxrwxr-x 3 es23btech11009 es23btech11009   4096 Nov 21 17:42 telemetry

Qwen3-32B paired 300-request training — 2026-11-21

SSH command (exit 0; key authentication, BatchMode):

```text
ssh -o BatchMode=yes -o ConnectTimeout=5 -o ConnectionAttempts=1 es23btech11009@192.168.50.147 'echo SSH_OK; for d in /home/es23btech11009/LLMServingSim/LLMServingSim/research/out/qwen32_single_300_uniform_four_faults /home/es23btech11009/LLMServingSim/LLMServingSim/research/out/qwen32_single_300_uniform_healthy; do echo DIR:$d; ls -l "$d"; done'
```

The command output above confirmed both directories and existing request_training_data.csv exports. Transfer commands, each exit 0 and no output:

```text
scp -r es23btech11009@192.168.50.147:/home/es23btech11009/LLMServingSim/LLMServingSim/research/out/qwen32_single_300_uniform_four_faults research/out/qwen32_single_300_uniform_training/
scp -r es23btech11009@192.168.50.147:/home/es23btech11009/LLMServingSim/LLMServingSim/research/out/qwen32_single_300_uniform_healthy research/out/qwen32_single_300_uniform_training/
```

ID audit command: `python research/out/inspect_300.py` (exit 0). Output: `four_faults unique_ids 300 minmax (0, 299)`; `healthy unique_ids 300 minmax (0, 299)`; `ids_equal True missing_healthy [] missing_faulty []`. Paired arrival, prompt length, and requested output length also matched for each request.

Training command: `python research/fault_detection/train_qwen32_uniform_three_models.py --data-root research/out/qwen32_single_300_uniform_training --out research/out/qwen32_single_300_uniform_training/model_results --seed 23` (exit 0). Exact console output:

```json
{
  "matched_ids": 300,
  "classifier": {
    "accuracy": 0.9333333333333333,
    "balanced_accuracy": 0.9333333333333333,
    "precision_faulty": 1.0,
    "recall_faulty": 0.8666666666666667,
    "f1_faulty": 0.9285714285714286,
    "roc_auc": 0.9877333333333334,
    "confusion_matrix_true_rows_healthy_faulty_pred_columns_healthy_faulty": [[75, 0], [10, 65]]
  },
  "localizer": "not_trained: valid three-class per-request labels unavailable",
  "regression": {
    "prefill": {"train_rows": 225, "test_rows": 75, "mae_ms": 1.6612943436084706, "rmse_ms": 1.9792415332118245, "r2": 0.9909457447951636, "mean_baseline_mae_ms": 18.592403394133335},
    "kv_transfer": {"train_rows": 225, "test_rows": 75, "mae_ms": 0.0, "rmse_ms": 0.0, "r2": 1.0, "mean_baseline_mae_ms": 0.0},
    "decode": {"train_rows": 225, "test_rows": 75, "mae_ms": 12.11323202878181, "rmse_ms": 18.7375703656339, "r2": 0.9999064567534836, "mean_baseline_mae_ms": 1756.3561093101632}
  }
}
```

No simulations were run. The faulty config has three transfer/network mechanisms plus a separate thermal throttle configuration; the directory's `four_faults` name is not four exclusive per-request classes. Communication ledger volume deltas are not exact timing counterfactuals. Healthy replay transfer intervals are all zero, so its transfer regression is a constant-target result.

## Corrected two-model implementation — 2026-11-21

The earlier exploratory run used workload/queue features for the classifier and reported 93.33% accuracy. The user clarified that the intended binary classifier should learn from observed stage slowdown telemetry, and selected only two models: the binary classifier and healthy-only stage regressors. Component localization was removed from the plan and code. The earlier classifier metric is superseded below.

### Schema normalization attempts (failures retained)

First command:

```text
python research/fault_detection/train_qwen32_uniform_three_models.py --data-root ..\LLMServingSim\research\out\qwen32_single_300_uniform_training --out ..\LLMServingSim\research\out\qwen32_single_300_uniform_training\model_results --seed 23
```

Exit code: 1. Exact error:

```text
KeyError: 'prefill_observed_ns'
```

After changing the feature name, the same command failed with exit code 1. Exact error:

```text
KeyError: 'prefill_service_observed_ns'
```

The healthy and faulty exporters use different prefill column names. The trainer now normalizes both to a shared observed-prefill feature before model construction.

### Reproducible two-model run

Command:

```text
python research/fault_detection/train_qwen32_binary_and_stage_regressors.py --data-root ..\LLMServingSim\research\out\qwen32_single_300_uniform_training --out ..\LLMServingSim\research\out\qwen32_single_300_uniform_training\model_results --seed 23
```

Exit code: 0. Exact output:

```json
{
  "matched_ids": 300,
  "classifier": {
    "accuracy": 1.0,
    "balanced_accuracy": 1.0,
    "precision_faulty": 1.0,
    "recall_faulty": 1.0,
    "f1_faulty": 1.0,
    "roc_auc": 1.0,
    "confusion_matrix_true_rows_healthy_faulty_pred_columns_healthy_faulty": [[75, 0], [0, 75]]
  },
  "regression": {
    "prefill": {"target": "prefill_service_observed_ns / 1e6 (ms)", "train_rows": 225, "test_rows": 75, "mae_ms": 1.6612943436084706, "rmse_ms": 1.9792415332118245, "r2": 0.9909457447951636, "mean_baseline_mae_ms": 18.592403394133335},
    "kv_transfer": {"target": "transfer_interval_observed_ns / 1e6 (ms)", "train_rows": 225, "test_rows": 75, "mae_ms": 0.0, "rmse_ms": 0.0, "r2": 1.0, "mean_baseline_mae_ms": 0.0},
    "decode": {"target": "decode_itl_sum_observed_ns / 1e6 (ms)", "train_rows": 225, "test_rows": 75, "mae_ms": 12.11323202878181, "rmse_ms": 18.7375703656339, "r2": 0.9999064567534836, "mean_baseline_mae_ms": 1756.3561093101632}
  }
}
```

Classifier inputs are measured per-request prefill, transfer/handoff, decode, TTFT, completion, queuing and ITL timing plus workload/queue context. These are legitimate observed telemetry features, not label or injection ground truth. The classifier therefore predicts completed-request healthy/faulty condition; it is retrospective. The regressors use only workload and pre-service queue inputs so target and completed timing telemetry cannot leak into regression. No component localizer is part of this version.

### Two-model artifact audit and instructor report

Verification command:

```text
python -c "import json,pathlib; p=pathlib.Path('..')/'LLMServingSim'/'research'/'out'/'qwen32_single_300_uniform_training'/'model_results'; s=json.loads((p/'training_summary.json').read_text()); r=s['regression']; print('models=',sorted(r)); print('classifier_features=',len(s['classifier']['features'])); print('regression_features=',len(r['features'])); print('stale_localizer_report=',(p/'localizer_report.json').exists()); print('accuracy=',s['classifier']['accuracy']); print('confusion=',s['classifier']['confusion_matrix_true_rows_healthy_faulty_pred_columns_healthy_faulty'])"
python -m py_compile research/fault_detection/train_qwen32_binary_and_stage_regressors.py
git diff --check
```

Exit code: 0. Exact output:

```text
models= ['counterfactual_caveat', 'counts', 'features', 'limitation', 'model', 'seed', 'split', 'stage_definitions', 'targets', 'training_condition']
classifier_features= 21
regression_features= 11
stale_localizer_report= False
accuracy= 1.0
confusion= [[75, 0], [0, 75]]
```

Created the instructor-facing project summary at repository root as `report.md`. Claude deck brief updated to selected two-model design and current held-out results. Component localization is not included in the selected implementation.
