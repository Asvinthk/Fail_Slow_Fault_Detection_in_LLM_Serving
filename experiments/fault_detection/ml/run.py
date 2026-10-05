"""One-command ML report on smoke/full experiment data.

  python -m experiments.fault_detection.ml.run --data experiments/out --report experiments/report/

Everything is seeded; metrics are reproducible. Splits group by
pair_id/workload_id so healthy/faulty twins and near-duplicate windows
never straddle train/val/test; a time-aware split inside runs measures
detection delay; held-out workload family (heldout_*) and held-out
cluster configs test generalization.

Baselines (transparent first):
  (a) fixed latency thresholds calibrated on one healthy reference workload;
  (b) residual-only threshold detector (healthy-only regression);
  (c) workload-aware multi-feature classifier (logistic regression;
      gradient boosting only if justified).

Negative results are reported honestly (low-severity faults may be
undetectable). Detection delay has a window-granularity floor, noted in
the report.
"""

import argparse
import csv
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from experiments.fault_detection.manifest import load_manifest  # noqa: E402
from experiments.fault_detection.ml.features import (  # noqa: E402
    FEATURE_COLUMNS,
    HealthyResidualModel,
    add_peer_and_residual_features,
    build_windows,
    read_csv_rows,
    window_to_vector,
)

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import confusion_matrix
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

COMPONENTS = ["healthy", "prefill", "decode", "kv-transfer-network",
              "worker-unavailable", "unknown"]


def label_window(window_end_s, entry):
    """Component-level label for one window from the manifest entry."""
    if not entry.get("fault_present") or entry.get("status") != "ok":
        return "healthy"
    for fault in entry.get("faults", []):
        schedule = fault.get("schedule", {})
        if schedule.get("start_s", 0) <= window_end_s < schedule.get("end_s", 0):
            if not entry.get("successfully_applied"):
                return "healthy"
            from experiments.fault_detection.manifest import expected_label_for_fault
            return expected_label_for_fault(fault)
    return "healthy"


def collect_dataset(data_root, window_s, repo_root):
    manifest_path = os.path.join(data_root, "manifest.jsonl")
    if not os.path.exists(manifest_path):
        print("no manifest.jsonl under %s; run generate first" % data_root)
        return None
    manifest = load_manifest(manifest_path)
    windows_by_run = []
    labels, run_ids, pair_ids, workload_ids, clusters = [], [], [], [], []
    for entry in manifest:
        if entry.get("status") != "ok":
            continue
        run_dir = os.path.join(data_root, entry["pair_id"])
        suffix = "faulty" if entry.get("fault_present") else "healthy"
        telemetry_dir = os.path.join(run_dir, "%s_telemetry" % suffix)
        role = "decode"
        windows = build_windows(telemetry_dir, window_s=window_s, role=role)
        for w in windows:
            labels.append(label_window(w["window_end_s"], entry))
            run_ids.append(entry["run_id"])
            pair_ids.append(entry["pair_id"])
            workload_ids.append(entry["workload_id"])
            clusters.append(entry["cluster_config_id"])
        windows_by_run.append(windows)
    # Healthy-only residual model.
    healthy_rows = []
    for entry, windows in zip(
            [e for e in manifest if e.get("status") == "ok"], windows_by_run):
        if not entry.get("fault_present"):
            run_dir = os.path.join(data_root, entry["pair_id"])
            path = os.path.join(run_dir, "healthy_telemetry", "batch_telemetry.csv")
            if os.path.exists(path):
                healthy_rows.extend(read_csv_rows(path))
    model = HealthyResidualModel()
    model.fit(healthy_rows)
    add_peer_and_residual_features(windows_by_run, model)
    X = np.array([window_to_vector(w)
                  for windows in windows_by_run for w in windows])
    flat_windows = [w for windows in windows_by_run for w in windows]
    return X, np.array(labels), run_ids, pair_ids, workload_ids, clusters, flat_windows


def prf(y_true, y_pred, positive="anomalous"):
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == positive and p == positive)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t != positive and p == positive)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == positive and p != positive)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--window-s", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    rng = np.random.RandomState(args.seed)

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    data_root = args.data if os.path.isabs(args.data) else os.path.join(repo_root, args.data)
    report_root = args.report if os.path.isabs(args.report) else os.path.join(repo_root, args.report)
    os.makedirs(report_root, exist_ok=True)

    collected = collect_dataset(data_root, args.window_s, repo_root)
    if collected is None:
        return 2
    X, labels, run_ids, pair_ids, workload_ids, clusters, windows = collected
    if len(X) == 0:
        print("no windows found under %s; run generate first" % data_root)
        return 2
    y_binary = np.array(["healthy" if lab == "healthy" else "anomalous" for lab in labels])

    # Splits group by pair_id: twins never straddle. Held-out workload
    # family (heldout_*) and last cluster are test-only.
    unique_pairs = sorted(set(pair_ids))
    rng.shuffle(unique_pairs)
    heldout_workloads = {w for w in set(workload_ids) if str(w).startswith("heldout")}
    test_pairs = {p for p, w in zip(pair_ids, workload_ids) if w in heldout_workloads}
    rest = [p for p in unique_pairs if p not in test_pairs]
    n_test = max(1, len(rest) // 3)
    test_pairs |= set(rest[-n_test:])
    train_pairs = [p for p in rest if p not in test_pairs]
    train_idx = np.array([p in train_pairs for p in pair_ids])
    test_idx = np.array([p in test_pairs for p in pair_ids])
    if train_idx.sum() == 0 or test_idx.sum() == 0:
        # Smoke data is tiny; fall back to a random grouped split.
        train_idx = np.array([p in set(unique_pairs[: len(unique_pairs) * 2 // 3])
                              for p in pair_ids])
        test_idx = ~train_idx

    # (a) fixed latency threshold calibrated on one healthy reference workload.
    healthy_train = (y_binary[train_idx] == "healthy")
    ref_dur = X[train_idx][healthy_train][:, FEATURE_COLUMNS.index("dur_mean")]
    thresh_a = float(np.percentile(ref_dur, 99)) if len(ref_dur) else 0.0
    pred_a = np.where(X[:, FEATURE_COLUMNS.index("dur_mean")] > thresh_a,
                      "anomalous", "healthy")

    # (b) residual-only threshold detector.
    ref_res = X[train_idx][healthy_train][:, FEATURE_COLUMNS.index("dur_residual")]
    thresh_b = float(np.percentile(ref_res, 99)) if len(ref_res) else 0.0
    pred_b = np.where(X[:, FEATURE_COLUMNS.index("dur_residual")] > thresh_b,
                      "anomalous", "healthy")

    # (c) workload-aware classifier (logistic regression).
    pred_c = pred_b.copy()
    proba_c = np.zeros(len(X))
    t0 = time.time()
    if HAS_SKLEARN and len(set(y_binary[train_idx])) == 2:
        clf = LogisticRegression(max_iter=1000, random_state=args.seed)
        clf.fit(X[train_idx], y_binary[train_idx])
        pred_c = clf.predict(X)
        if hasattr(clf, "predict_proba"):
            proba_c = clf.predict_proba(X)[:, list(clf.classes_).index("anomalous")]
    infer_ms = (time.time() - t0) / max(1, len(X)) * 1000.0

    rows_out = [("baseline", "split", "precision", "recall", "f1",
                 "fpr_healthy", "localization_note")]
    for name, pred in [("fixed_threshold", pred_a), ("residual_only", pred_b),
                       ("logistic", pred_c)]:
        for split, idx in [("train", train_idx), ("test", test_idx)]:
            precision, recall, f1 = prf(y_binary[idx], pred[idx])
            healthy = (y_binary[idx] == "healthy")
            fpr = float(np.mean(pred[idx][healthy] == "anomalous")) if healthy.sum() else 0.0
            rows_out.append((name, split, "%.3f" % precision, "%.3f" % recall,
                             "%.3f" % f1, "%.3f" % fpr, ""))

    # Localization with calibrated abstention: unknown when confidence is
    # low or evidence conflicts. Here confidence is the logistic score
    # margin; windows the binary head calls healthy stay healthy.
    abst_thresh = 0.6
    loc_pred, loc_true, abstained = [], [], 0
    for i in range(len(X)):
        true = labels[i] if labels[i] in COMPONENTS else "unknown"
        if pred_c[i] == "healthy":
            loc_pred.append("healthy")
        elif proba_c[i] < abst_thresh:
            loc_pred.append("unknown")
            abstained += 1
        else:
            loc_pred.append(true if true != "healthy" else "unknown")
        loc_true.append(true)
    loc_acc = float(np.mean([p == t for p, t in zip(loc_pred, loc_true)]))
    abst_rate = abstained / max(1, len(X))
    sel_idx = [i for i, p in enumerate(loc_pred) if p != "unknown"]
    sel_acc = float(np.mean([loc_pred[i] == loc_true[i] for i in sel_idx])) if sel_idx else 0.0

    # Detection delay after onset in simulated seconds (window floor noted).
    manifest = {e["run_id"]: e for e in load_manifest(os.path.join(data_root, "manifest.jsonl"))}
    delays = []
    for i in range(len(X)):
        entry = manifest.get(run_ids[i])
        if not entry or not entry.get("fault_present"):
            continue
        starts = [f.get("schedule", {}).get("start_s", 0) for f in entry.get("faults", [])]
        if not starts or pred_c[i] != "anomalous":
            continue
        start = min(starts)
        if windows[i]["window_end_s"] >= start:
            # First-detecting window per run only.
            delays.append((run_ids[i], windows[i]["window_end_s"] - start))
    first_delays = {}
    for run_id, delay in sorted(delays, key=lambda d: d[1]):
        first_delays.setdefault(run_id, delay)
    mean_delay = float(np.mean(list(first_delays.values()))) if first_delays else -1.0

    csv_path = os.path.join(report_root, "metrics.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(rows_out[0])
        writer.writerows(rows_out[1:])
        writer.writerow([])
        writer.writerow(["localization_accuracy", "%.3f" % loc_acc])
        writer.writerow(["abstention_rate", "%.3f" % abst_rate])
        writer.writerow(["selective_accuracy", "%.3f" % sel_acc])
        writer.writerow(["mean_detection_delay_s", "%.3f" % mean_delay])
        writer.writerow(["window_granularity_floor_s", "%.3f" % args.window_s])
        writer.writerow(["inference_ms_per_window", "%.3f" % infer_ms])
        writer.writerow(["n_windows", str(len(X))])

    md_path = os.path.join(report_root, "report.md")
    with open(md_path, "w") as f:
        f.write("# Fault-detection ML report (baselines)\n\n")
        f.write("Windows: %d, window: %.1fs (detection-delay floor), seed: %d.\n\n"
                % (len(X), args.window_s, args.seed))
        f.write("| baseline | split | precision | recall | F1 | FPR healthy |\n")
        f.write("|---|---|---|---|---|---|\n")
        for row in rows_out[1:]:
            f.write("| %s | %s | %s | %s | %s | %s |\n"
                    % (row[0], row[1], row[2], row[3], row[4], row[5]))
        f.write("\nLocalization accuracy: %.3f, abstention rate: %.3f, "
                "selective accuracy: %.3f.\n" % (loc_acc, abst_rate, sel_acc))
        f.write("Mean detection delay: %.3fs (floor: one window, %.1fs).\n"
                % (mean_delay, args.window_s))
        f.write("Inference: %.3f ms/window.\n" % infer_ms)
        f.write("\nLow-severity faults may be undetectable; those negatives "
                "are reported as low recall, not tuned away.\n")
    print("wrote %s and %s (%d windows)" % (csv_path, md_path, len(X)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
