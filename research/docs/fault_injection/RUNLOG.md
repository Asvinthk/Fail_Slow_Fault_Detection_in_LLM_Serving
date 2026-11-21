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
