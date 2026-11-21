"""Train binary healthy/faulty and healthy-only stage-time models."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             confusion_matrix, f1_score, mean_absolute_error,
                             mean_squared_error, precision_score, r2_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


FEATURES = [
    "input_toks", "output_toks", "arrival_gap_ns", "route_lag_ns",
    "prefill_waiting_before", "prefill_running_before",
    "total_prefill_waiting_before", "total_prefill_running_before",
    "total_decode_waiting_before", "total_decode_running_before",
    "pending_unrouted_before", "prefill_service_observed_ns",
    "transfer_interval_observed_ns", "decode_itl_sum_observed_ns",
    "ttft_ns", "latency_ns", "queuing_delay_ns", "tpot_ns",
    "itl_mean_ns", "itl_p99_ns", "itl_max_ns",
]
STAGES = {
    "prefill": "prefill_service_observed_ns",
    "kv_transfer": "transfer_interval_observed_ns",
    "decode": "decode_itl_sum_observed_ns",
}
FORBIDDEN_NAMES = ("label", "fault_id", "audit", "added", "delta",
                   "applied", "nominal", "ledger", "manifest", "absolute")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def safe_features(run_dir: Path) -> list[dict]:
    table = {int(r["request_id"]): r for r in read_csv(run_dir / "request_training_data.csv")}
    qrows = {int(r["request_id"]): r for r in read_csv(
        run_dir / "telemetry" / "arrival_queue_telemetry.csv")}
    ids = sorted(table)
    previous_arrival = None
    result = []
    for rid in ids:
        r, q = table[rid], qrows[rid]
        arrival = int(r["arrival_ns"])
        vals = {
            "input_toks": float(r["input_toks"]),
            "output_toks": float(r["output_toks"]),
            "arrival_gap_ns": float(arrival - previous_arrival) if previous_arrival is not None else 0.0,
            "route_lag_ns": float(q["route_lag_ns"]),
        }
        previous_arrival = arrival
        for key in FEATURES[4:11]:
            vals[key] = float(q[key])
        # The healthy and faulty exporters used different names for the same
        # observed prefill interval; normalize them to one feature name.
        aliases = {
            "prefill_service_observed_ns": ("prefill_service_observed_ns", "prefill_observed_ns"),
            "transfer_interval_observed_ns": ("transfer_interval_observed_ns",),
            "decode_itl_sum_observed_ns": ("decode_itl_sum_observed_ns",),
        }
        for name, candidates in aliases.items():
            value = next((r[c] for c in candidates if c in r), None)
            if value is None:
                raise ValueError(f"{run_dir.name} request {rid} missing observed feature {name}")
            vals[name] = float(value)
        for name, raw_name in (("ttft_ns", "ttft_ns"), ("latency_ns", "latency_ns"),
                               ("queuing_delay_ns", "queuing_delay_ns"),
                               ("tpot_ns", "tpot"), ("itl_mean_ns", "itl_mean_ns"),
                               ("itl_p99_ns", "itl_p99_ns"), ("itl_max_ns", "itl_max_ns")):
            vals[name] = float(r[raw_name])
        result.append({"request_id": rid, "arrival_ns": arrival,
                       "features": vals, "raw": r})
    return result


def matrix(rows: list[dict], features: list[str] = FEATURES) -> np.ndarray:
    return np.asarray([[r["features"][name] for name in features] for r in rows], dtype=float)


def assert_no_leakage() -> None:
    bad = [f for f in FEATURES if any(term in f.lower() for term in FORBIDDEN_NAMES)]
    if bad:
        raise AssertionError(f"forbidden feature names: {bad}")


def classifier(healthy: list[dict], faulty: list[dict], output: Path, seed: int) -> dict:
    rows = healthy + faulty
    X = matrix(rows)
    y = np.asarray([0] * len(healthy) + [1] * len(faulty), dtype=int)
    pair_groups = np.asarray([r["request_id"] for r in healthy + faulty])
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed)
    train_idx, test_idx = next(splitter.split(X, y, pair_groups))
    model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                          LogisticRegression(max_iter=3000, random_state=seed))
    model.fit(X[train_idx], y[train_idx])
    pred = model.predict(X[test_idx])
    prob = model.predict_proba(X[test_idx])[:, 1]
    cm = confusion_matrix(y[test_idx], pred, labels=[0, 1])
    report = {
        "model": "logistic regression; median imputation + standard scaling fitted on training partition",
        "target": "condition membership: healthy replay=0, combined-fault run=1; synthetic healthy-versus-faulty label, not an independently calibrated SLO threshold",
        "features": FEATURES,
        "counts": {"healthy": len(healthy), "combined_fault": len(faulty),
                   "train": len(train_idx), "test": len(test_idx),
                   "test_healthy": int((y[test_idx] == 0).sum()),
                   "test_faulty": int((y[test_idx] == 1).sum())},
        "split": "paired request IDs grouped together; 75/25 GroupShuffleSplit",
        "seed": seed,
        "accuracy": float(accuracy_score(y[test_idx], pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y[test_idx], pred)),
        "precision_faulty": float(precision_score(y[test_idx], pred, zero_division=0)),
        "recall_faulty": float(recall_score(y[test_idx], pred, zero_division=0)),
        "f1_faulty": float(f1_score(y[test_idx], pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y[test_idx], prob)),
        "confusion_matrix_true_rows_healthy_faulty_pred_columns_healthy_faulty": cm.tolist(),
        "feature_timing": "uses directly observed per-request prefill, transfer/handoff, and decode timing telemetry; prediction is made after request timing is available",
        "caveat": "single matched workload and topology; post-request observable timing features make this completed-request healthy-versus-faulty classifier, not an early-warning or unseen-workload evaluation",
    }
    joblib.dump({"model": model, "features": FEATURES, "report": report}, output / "classifier.joblib")
    (output / "classifier_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def regression(healthy: list[dict], output: Path, seed: int) -> dict:
    # Use only request-time workload/queue features. Stage targets and other
    # completed-request timing telemetry must never predict themselves.
    regression_features = FEATURES[:11]
    X = matrix(healthy, regression_features)
    arrivals = np.asarray([r["arrival_ns"] for r in healthy])
    # Temporal holdout preserves causal ordering and avoids near-duplicate
    # neighboring arrivals leaking across partitions.
    order = np.argsort(arrivals)
    split = int(len(order) * 0.75)
    train_idx, test_idx = order[:split], order[split:]
    targets = {}
    for stage, column in STAGES.items():
        y = np.asarray([float(r["raw"][column]) / 1e6 for r in healthy])
        model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LinearRegression())
        model.fit(X[train_idx], y[train_idx])
        pred = model.predict(X[test_idx])
        baseline = np.full_like(y[test_idx], float(np.mean(y[train_idx])))
        targets[stage] = {
            "target": column + " / 1e6 (ms)",
            "train_rows": int(len(train_idx)), "test_rows": int(len(test_idx)),
            "mae_ms": float(mean_absolute_error(y[test_idx], pred)),
            "rmse_ms": float(np.sqrt(mean_squared_error(y[test_idx], pred))),
            "r2": float(r2_score(y[test_idx], pred)),
            "mean_baseline_mae_ms": float(mean_absolute_error(y[test_idx], baseline)),
        }
        joblib.dump({"model": model, "features": regression_features, "target": column},
                    output / f"{stage}_regressor.joblib")
    report = {
        "model": "linear regression; median imputation + standard scaling fitted on training partition",
        "training_condition": "healthy replay only",
        "features": regression_features,
        "counts": {"healthy_total": len(healthy), "train": int(len(train_idx)), "test": int(len(test_idx))},
        "split": "chronological 75/25 by request arrival order", "seed": seed,
        "targets": targets,
        "stage_definitions": {
            "prefill": "first_scheduled_ns to prefill_done_ns; queue-free prefill service interval",
            "kv_transfer": "decode_admit_ns minus prefill_done_ns, the measured simulator handoff interval",
            "decode": "mean ITL multiplied by ITL count; aggregate decode ITL sum proxy",
        },
        "counterfactual_caveat": "healthy replay timings are measured healthy targets; they are not exact per-request healthy communication times subtracted from the faulty run",
        "limitation": "regression is a baseline prototype only; no residual threshold or alarm is inferred here",
    }
    (output / "regression_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=23)
    args = parser.parse_args()
    assert_no_leakage()
    healthy_dir = args.data_root / "qwen32_single_300_uniform_healthy"
    faulty_dir = args.data_root / "qwen32_single_300_uniform_four_faults"
    healthy, faulty = safe_features(healthy_dir), safe_features(faulty_dir)
    h_ids, f_ids = {r["request_id"] for r in healthy}, {r["request_id"] for r in faulty}
    if len(healthy) != 300 or len(faulty) != 300 or h_ids != f_ids:
        raise ValueError(f"expected 300 matched unique IDs; healthy={len(h_ids)}, faulty={len(f_ids)}")
    # Validate matching workload covariates and arrivals before pooling.
    h_by, f_by = ({r["request_id"]: r for r in rows} for rows in (healthy, faulty))
    for rid in sorted(h_ids):
        for key in ("arrival_ns", "input_toks", "output_toks"):
            if str(h_by[rid]["raw"][key]) != str(f_by[rid]["raw"][key]):
                raise ValueError(f"paired request {rid} differs at {key}")
    args.out.mkdir(parents=True, exist_ok=True)
    stale_localizer_report = args.out / "localizer_report.json"
    if stale_localizer_report.exists():
        stale_localizer_report.unlink()
    summary = {
        "paired_ids": 300, "id_min": min(h_ids), "id_max": max(h_ids),
        "arrival_and_prompt_output_lengths_match": True,
        "data_sources": {"healthy": str(healthy_dir), "combined_fault": str(faulty_dir)},
        "classifier": classifier(healthy, faulty, args.out, args.seed),
        "regression": regression(healthy, args.out, args.seed),
        "leakage_controls": {
            "excluded": ["request_id", "instance IDs", "absolute timestamps", "labels",
                         "fault IDs/config", "audit/ledger/manifest", "added-time and compute-delta fields"],
            "classifier_features_are": "workload/queue covariates plus directly observed stage/completion timings; no injection truth or added-time fields",
            "regression_features_are": "workload/queue covariates only; observed timing targets excluded from predictors",
            "regression_fit_data": "healthy only",
        },
    }
    (args.out / "training_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"matched_ids": 300,
                      "classifier": {k: summary["classifier"][k] for k in
                                     ("accuracy", "balanced_accuracy", "precision_faulty", "recall_faulty", "f1_faulty", "roc_auc", "confusion_matrix_true_rows_healthy_faulty_pred_columns_healthy_faulty")},
                      "regression": summary["regression"]["targets"]}, indent=2))


if __name__ == "__main__":
    main()
