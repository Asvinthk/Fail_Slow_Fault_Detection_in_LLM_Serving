"""One-command ML report on smoke/full experiment data.

  python -m experiments.fault_detection.ml.run --data experiments/out --report experiments/report/

Rewritten after external review (see report.md section 16). What changed
and why:

  * labels come from the injection AUDIT, per window and per instance
    (labels.py), not from the fault schedule plus a run-level flag. A
    window is a positive only where the mechanism actually applied to
    that instance (F3);
  * roles are resolved per instance, from the telemetry itself, so
    prefill windows are no longer compared against decode windows (F2);
  * the transparent baselines use robust thresholds (median + k*MAD)
    calibrated on healthy TRAIN windows, because with ~14 healthy
    windows a "p99" is just the maximum (N4);
  * an EWMA-residual detector is added; it accumulates evidence across
    windows instead of testing each window independently (N4);
  * splits are grouped by RUN (pair), never by window, and every
    proportion carries a Wilson interval; P/R/F1 also get a bootstrap
    interval over runs (N5);
  * downstream, recovering and neutral windows are excluded from the
    metric and reported separately, so a slow decode backing up prefill
    is not scored as a false alarm (N7/N8/Q10);
  * localisation is a real per-instance residual ranking with
    abstention, scored top-1/top-3 against `decode:1`-style labels --
    the previous number echoed ground truth (F3);
  * detection delay is measured on a 0.5 s sliding stride, so the
    reported delay is not floored at the 2 s evaluation window (Q11);
  * overhead is measured on predict only, not on model fitting (F10);
  * logistic regression is opt-in (--with-logistic): with a few dozen
    windows it cannot be defended as a baseline (Q6).

Negative results are reported honestly: low-severity faults may be
undetectable, and that shows up as low recall rather than being tuned
away.
"""

import argparse
import csv
import json
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from experiments.fault_detection.labels import (  # noqa: E402
    EXCLUDED_KINDS,
    NEGATIVE_KINDS,
    POSITIVE_KINDS,
    attach_labels,
    label_windows,
)
from experiments.fault_detection.manifest import load_manifest  # noqa: E402
from experiments.fault_detection.ml.features import (  # noqa: E402
    FEATURE_COLUMNS,
    HealthyResidualModel,
    _as_float,
    _dedupe_batches,
    add_peer_and_residual_features,
    build_windows,
    infer_instance_roles,
    read_csv_rows,
    window_to_vector,
)

try:
    from sklearn.linear_model import LogisticRegression
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

DELAY_STRIDE_S = 0.5
MAD_K = 4.0
EWMA_ALPHA = 0.5


# --------------------------------------------------------------------------
# statistics helpers
# --------------------------------------------------------------------------

def wilson(k, n, z=1.96):
    """Wilson score interval for a proportion (k of n)."""
    if n <= 0:
        return 0.0, 0.0
    p = k / float(n)
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2.0 * n)) / denom
    half = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def mad_threshold(values, k=MAD_K):
    """Robust upper threshold: median + k * 1.4826 * MAD.

    A p99 over a handful of healthy windows IS the maximum, so a single
    benign spike sets the threshold for everything (N4). MAD is
    break-down-robust. When the healthy set has zero spread (no timing
    jitter), the floor keeps the threshold just above the median instead
    of collapsing onto it.
    """
    values = [v for v in values if v is not None and np.isfinite(v)]
    if not values:
        return float("inf")
    arr = np.array(values, dtype=float)
    med = float(np.median(arr))
    mad = float(np.median(np.abs(arr - med)))
    sigma = 1.4826 * mad
    if sigma <= 0:
        sigma = float(np.std(arr))
    floor = 0.01 * abs(med) + 1.0
    return med + max(k * sigma, floor)


def prf_counts(y_true, y_pred):
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    return tp, fp, fn, tn


def prf_from_counts(tp, fp, fn, tn):
    precision = tp / float(tp + fp) if (tp + fp) else 0.0
    recall = tp / float(tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    fpr = fp / float(fp + tn) if (fp + tn) else 0.0
    return precision, recall, f1, fpr


def bootstrap_over_runs(runs, pred_by_run, true_by_run, n_boot=1000, seed=42):
    """Percentile CI for P/R/F1/FPR, resampling RUNS (not windows)."""
    rng = np.random.RandomState(seed)
    unique = sorted(runs)
    if len(unique) < 2:
        return None
    stats = []
    for _ in range(n_boot):
        pick = rng.choice(len(unique), size=len(unique), replace=True)
        tp = fp = fn = tn = 0
        for idx in pick:
            run = unique[idx]
            t = true_by_run[run]
            p = pred_by_run[run]
            a, b, c, d = prf_counts(t, p)
            tp += a
            fp += b
            fn += c
            tn += d
        stats.append(prf_from_counts(tp, fp, fn, tn))
    arr = np.array(stats)
    return [tuple(np.percentile(arr[:, i], [2.5, 97.5])) for i in range(4)]


# --------------------------------------------------------------------------
# dataset
# --------------------------------------------------------------------------

def collect_dataset(data_root, window_s, applied_share, recovery_s):
    """Load every run, build windows, attach audit-derived labels."""
    manifest_path = os.path.join(data_root, "manifest.jsonl")
    if not os.path.exists(manifest_path):
        print("no manifest.jsonl under %s; run generate first" % data_root)
        return None
    manifest = [e for e in load_manifest(manifest_path) if e.get("status") == "ok"]
    windows_by_run = []
    rows = []
    missing_audit = []
    for entry in manifest:
        run_dir = os.path.join(data_root, entry["pair_id"])
        suffix = "faulty" if entry.get("fault_present") else "healthy"
        telemetry_dir = os.path.join(run_dir, "%s_telemetry" % suffix)
        if not os.path.exists(os.path.join(telemetry_dir, "batch_telemetry.csv")):
            continue
        roles = infer_instance_roles(telemetry_dir)
        windows = build_windows(telemetry_dir, window_s=window_s, roles=roles)
        labels = label_windows(telemetry_dir, entry, window_s,
                               applied_share=applied_share,
                               recovery_s=recovery_s)
        if entry.get("fault_present") and not any(
                lab["kind"] in POSITIVE_KINDS + EXCLUDED_KINDS
                for lab in labels.values()):
            # No audit activity at all: the run cannot be scored, and
            # silently calling it healthy would inflate the FPR
            # denominator. Report it.
            missing_audit.append(entry["run_id"])
        windows = attach_labels(windows, labels)
        for w in windows:
            w["run_id"] = entry["run_id"]
            w["pair_id"] = entry["pair_id"]
            w["workload_id"] = entry["workload_id"]
            w["cluster_config_id"] = entry["cluster_config_id"]
        windows_by_run.append(windows)
        rows.extend(windows)

    # Healthy-only residual model, fitted on raw healthy batches.
    healthy_rows = []
    for entry in manifest:
        if entry.get("fault_present"):
            continue
        path = os.path.join(data_root, entry["pair_id"], "healthy_telemetry",
                            "batch_telemetry.csv")
        if os.path.exists(path):
            healthy_rows.extend(read_csv_rows(path))
    model = HealthyResidualModel()
    fitted = model.fit(healthy_rows)
    add_peer_and_residual_features(windows_by_run, model)
    return {
        "rows": rows,
        "windows_by_run": windows_by_run,
        "model": model,
        "model_fitted": fitted,
        "manifest": {e["run_id"]: e for e in manifest},
        "missing_audit": missing_audit,
    }


def eligible_rows(rows):
    """Rows that enter the detection metric (root_cause vs unaffected)."""
    return [r for r in rows
            if r["label_kind"] in POSITIVE_KINDS or r["label_kind"] in NEGATIVE_KINDS]


# --------------------------------------------------------------------------
# detectors
# --------------------------------------------------------------------------

def role_thresholds(rows, key, k=MAD_K):
    per_role = {}
    for role in sorted({r["role"] for r in rows}):
        values = [r[key] for r in rows
                  if r["role"] == role and r["label_kind"] in NEGATIVE_KINDS]
        per_role[role] = mad_threshold(values, k)
    return per_role


def ewma_series(rows, key, alpha=EWMA_ALPHA):
    """EWMA of `key` per (run, instance), ordered by window end."""
    series = {}
    by_key = {}
    for r in rows:
        by_key.setdefault((r["run_id"], r["instance_id"]), []).append(r)
    for k, items in by_key.items():
        items.sort(key=lambda x: x["window_end_s"])
        value = None
        for item in items:
            x = item[key]
            value = x if value is None else alpha * x + (1.0 - alpha) * value
            series[(k, item["window_end_s"])] = value
    return series


def predict_series(rows, series, thresholds, key):
    pred = np.zeros(len(rows), dtype=int)
    for i, r in enumerate(rows):
        value = series.get(((r["run_id"], r["instance_id"]), r["window_end_s"]))
        if value is None:
            continue
        pred[i] = 1 if value > thresholds.get(r["role"], float("inf")) else 0
    return pred


# --------------------------------------------------------------------------
# localisation
# --------------------------------------------------------------------------

def localisation(rows, thresholds, key="dur_residual"):
    """Rank instances by residual inside each window; score top-1/top-3.

    Only root_cause windows are scored. The candidate set is EVERY
    instance present in that window (including healthy ones), which is
    the realistic task: the localiser does not know how many faults
    exist. Abstention is allowed when no instance exceeds its role's
    healthy-calibrated threshold.
    """
    by_window = {}
    for r in rows:
        by_window.setdefault((r["run_id"], r["window_end_s"]), []).append(r)
    top1 = top3 = scored = abstained = sel_correct = sel_total = 0
    for (_run, _end), items in by_window.items():
        positives = [r for r in items if r["label_kind"] in POSITIVE_KINDS]
        if not positives:
            continue
        ranked = sorted(items, key=lambda r: r[key], reverse=True)
        best = ranked[0]
        for true in positives:
            scored += 1
            if best["instance_id"] == true["label_instance_id"]:
                top1 += 1
            if true["label_instance_id"] in [r["instance_id"] for r in ranked[:3]]:
                top3 += 1
        if best[key] <= thresholds.get(best["role"], float("inf")):
            abstained += 1
            continue
        for true in positives:
            sel_total += 1
            if best["instance_id"] == true["label_instance_id"]:
                sel_correct += 1
    return {
        "scored": scored,
        "top1": top1 / float(scored) if scored else 0.0,
        "top3": top3 / float(scored) if scored else 0.0,
        "abstained_windows": abstained,
        "coverage": 1.0 - abstained / float(len(by_window)) if by_window else 0.0,
        "selective_top1": sel_correct / float(sel_total) if sel_total else 0.0,
    }


# --------------------------------------------------------------------------
# detection delay (0.5 s sliding stride)
# --------------------------------------------------------------------------

def batch_residuals(telemetry_dir, model):
    """Per-batch residual using the healthy shape model (observable only)."""
    path = os.path.join(telemetry_dir, "batch_telemetry.csv")
    if not os.path.exists(path):
        return []
    out = []
    for row, _mult in _dedupe_batches(read_csv_rows(path)):
        if model.coef_ is None:
            continue
        x = np.array([1.0, _as_float(row, "total_len"),
                      _as_float(row, "num_prefill"), _as_float(row, "n_decode"),
                      _as_float(row, "kv_len")])
        residual = _as_float(row, "duration_ns") - float(x @ model.coef_)
        out.append({
            "finish_ns": _as_float(row, "finish_ns"),
            "instance_id": int(_as_float(row, "instance_id")),
            "residual": residual,
        })
    out.sort(key=lambda b: b["finish_ns"])
    return out


def detection_delay(dataset, data_root, window_s, thresholds):
    """First EWMA crossing after onset, at 0.5 s stride.

    Delay is reported in simulated seconds AND in requests completed
    between onset and the alarm (review Q11), because a window-granular
    delay is an artefact of the window length.
    """
    model = dataset["model"]
    results = []
    for run_id, entry in dataset["manifest"].items():
        if not entry.get("fault_present"):
            continue
        starts = [f.get("schedule", {}).get("start_s") for f in entry.get("faults", [])]
        starts = [s for s in starts if s is not None]
        if not starts:
            continue
        onset = min(starts)
        run_dir = os.path.join(data_root, entry["pair_id"])
        suffix = "faulty" if entry.get("fault_present") else "healthy"
        telemetry_dir = os.path.join(run_dir, "%s_telemetry" % suffix)
        batches = batch_residuals(telemetry_dir, model)
        if not batches:
            continue
        req_path = os.path.join(telemetry_dir, "request_telemetry.csv")
        ends = []
        if os.path.exists(req_path):
            for r in read_csv_rows(req_path):
                ends.append(_as_float(r, "arrival_ns") + _as_float(r, "latency_ns"))
        ends.sort()
        horizon = batches[-1]["finish_ns"] / 1e9
        # EWMA warm-started from the beginning of the run, per instance.
        ewma = {}
        idx = 0
        alarm_s = None
        alarm_requests = 0
        t = 0.0
        while t <= horizon + 1e-9:
            limit = t * 1e9
            while idx < len(batches) and batches[idx]["finish_ns"] <= limit:
                b = batches[idx]
                prev = ewma.get(b["instance_id"])
                ewma[b["instance_id"]] = b["residual"] if prev is None \
                    else EWMA_ALPHA * b["residual"] + (1.0 - EWMA_ALPHA) * prev
                idx += 1
            if t >= onset and alarm_s is None:
                for inst, value in ewma.items():
                    # Role of the instance is not needed here: use the
                    # loosest threshold so the delay is not overstated by
                    # a role mismatch.
                    limit_thresh = min(thresholds.values()) if thresholds else float("inf")
                    if value > limit_thresh:
                        alarm_s = t
                        alarm_requests = sum(1 for e in ends if onset * 1e9 < e <= limit)
                        break
            t += DELAY_STRIDE_S
        if alarm_s is not None:
            results.append({"run_id": run_id, "delay_s": alarm_s - onset,
                            "delay_requests": alarm_requests})
    if not results:
        return {"n_runs": 0, "mean_s": -1.0, "median_s": -1.0,
                "mean_requests": -1.0, "details": []}
    delays = [r["delay_s"] for r in results]
    return {
        "n_runs": len(results),
        "mean_s": float(np.mean(delays)),
        "median_s": float(np.median(delays)),
        "mean_requests": float(np.mean([r["delay_requests"] for r in results])),
        "details": results,
    }


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--window-s", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--applied-share", type=float, default=0.5,
                        help="share of an instance's batches that must carry an applied audit "
                             "record for the window to be a root-cause positive")
    parser.add_argument("--recovery-s", type=float, default=2.0,
                        help="windows this long after a fault ends are 'recovering' (excluded)")
    parser.add_argument("--with-logistic", action="store_true",
                        help="also report logistic regression (not defensible at small n; off by default)")
    parser.add_argument("--no-bootstrap", action="store_true")
    args = parser.parse_args()

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    data_root = args.data if os.path.isabs(args.data) else os.path.join(repo_root, args.data)
    report_root = args.report if os.path.isabs(args.report) else os.path.join(repo_root, args.report)
    os.makedirs(report_root, exist_ok=True)

    dataset = collect_dataset(data_root, args.window_s, args.applied_share,
                              args.recovery_s)
    if dataset is None:
        return 2
    rows = dataset["rows"]
    if not rows:
        print("no windows found under %s; run generate first" % data_root)
        return 2

    rows = eligible_rows(rows) if any(
        r["label_kind"] in POSITIVE_KINDS for r in rows) else rows
    if not rows:
        print("no scoreable windows under %s (no audit activity?)" % data_root)
        return 2
    all_rows = dataset["rows"]

    # ---- run-level split (twins and windows never straddle) -------------
    pairs = sorted({r["pair_id"] for r in rows})
    rng = np.random.RandomState(args.seed)
    heldout = {r["pair_id"] for r in rows if str(r["workload_id"]).startswith("heldout")}
    rest = [p for p in pairs if p not in heldout]
    rng.shuffle(rest)
    n_test = max(1, len(rest) // 3) if len(rest) > 1 else 0
    test_pairs = heldout | set(rest[-n_test:]) if n_test else heldout
    train_pairs = {p for p in pairs if p not in test_pairs}
    if not train_pairs or not test_pairs:
        # Tiny smoke data: split runs 2/3 - 1/3 deterministically.
        train_pairs = set(pairs[: max(1, len(pairs) * 2 // 3)])
        test_pairs = set(pairs) - train_pairs
    is_train = np.array([r["pair_id"] in train_pairs for r in rows])
    is_test = ~is_train

    # ---- thresholds calibrated on HEALTHY TRAIN windows only ------------
    train_rows = [r for r, tr in zip(rows, is_train) if tr]
    thr_dur = role_thresholds(train_rows, "dur_mean")
    thr_res = role_thresholds(train_rows, "dur_residual")

    # ---- detectors -------------------------------------------------------
    dur = np.array([r["dur_mean"] for r in rows])
    res = np.array([r["dur_residual"] for r in rows])
    roles = [r["role"] for r in rows]

    pred_a = np.array([1 if dur[i] > thr_dur.get(roles[i], float("inf")) else 0
                       for i in range(len(rows))])
    pred_b = np.array([1 if res[i] > thr_res.get(roles[i], float("inf")) else 0
                       for i in range(len(rows))])

    # EWMA on the residual, per (run, instance), predict-only timing.
    t0 = time.time()
    series = ewma_series(rows, "dur_residual")
    pred_c = predict_series(rows, series, thr_res, "dur_residual")
    infer_ms = (time.time() - t0) / max(1, len(rows)) * 1000.0

    pred_d = None
    if args.with_logistic and HAS_SKLEARN:
        y_all = np.array([1 if r["label_kind"] in POSITIVE_KINDS else 0 for r in rows])
        X = np.array([window_to_vector(r) for r in rows])
        if len(set(y_all[is_train])) == 2:
            clf = LogisticRegression(max_iter=1000, random_state=args.seed)
            clf.fit(X[is_train], y_all[is_train])
            pred_d = clf.predict(X)

    y = np.array([1 if r["label_kind"] in POSITIVE_KINDS else 0 for r in rows])

    baselines = [("fixed_threshold_robust", pred_a),
                 ("residual_robust", pred_b),
                 ("ewma_residual", pred_c)]
    if pred_d is not None:
        baselines.append(("logistic", pred_d))

    # ---- metrics with CIs -------------------------------------------------
    table = []
    for name, pred in baselines:
        for split, mask in (("train", is_train), ("test", is_test)):
            tp, fp, fn, tn = prf_counts(y[mask], pred[mask])
            precision, recall, f1, fpr = prf_from_counts(tp, fp, fn, tn)
            rec_lo, rec_hi = wilson(tp, tp + fn)
            fpr_lo, fpr_hi = wilson(fp, fp + tn)
            ci = None
            if not args.no_bootstrap:
                run_ids = [r["pair_id"] for r, m in zip(rows, mask) if m]
                pred_by_run, true_by_run = {}, {}
                for r, p, t, m in zip(rows, pred, y, mask):
                    if not m:
                        continue
                    pred_by_run.setdefault(r["pair_id"], []).append(p)
                    true_by_run.setdefault(r["pair_id"], []).append(t)
                pred_by_run = {k: np.array(v) for k, v in pred_by_run.items()}
                true_by_run = {k: np.array(v) for k, v in true_by_run.items()}
                ci = bootstrap_over_runs(run_ids, pred_by_run, true_by_run,
                                         seed=args.seed)
            table.append({
                "baseline": name, "split": split,
                "precision": precision, "recall": recall, "f1": f1, "fpr": fpr,
                "tp": tp, "fp": fp, "fn": fn, "tn": tn,
                "recall_ci": (rec_lo, rec_hi), "fpr_ci": (fpr_lo, fpr_hi),
                "boot_ci": ci,
            })

    # ---- excluded-kind accounting ----------------------------------------
    kind_counts = {}
    for r in all_rows:
        kind_counts[r["label_kind"]] = kind_counts.get(r["label_kind"], 0) + 1

    loc = localisation(rows, thr_res)
    delay = detection_delay(dataset, data_root, args.window_s, thr_res)

    # ---- write metrics.csv ------------------------------------------------
    csv_path = os.path.join(report_root, "metrics.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["baseline", "split", "precision", "recall", "f1",
                         "fpr_healthy", "tp", "fp", "fn", "tn",
                         "recall_ci_lo", "recall_ci_hi",
                         "fpr_ci_lo", "fpr_ci_hi", "localization_note"])
        for row in table:
            writer.writerow([
                row["baseline"], row["split"],
                "%.3f" % row["precision"], "%.3f" % row["recall"],
                "%.3f" % row["f1"], "%.3f" % row["fpr"],
                row["tp"], row["fp"], row["fn"], row["tn"],
                "%.3f" % row["recall_ci"][0], "%.3f" % row["recall_ci"][1],
                "%.3f" % row["fpr_ci"][0], "%.3f" % row["fpr_ci"][1], ""])
        writer.writerow([])
        for key, value in [
                ("localization_top1", "%.3f" % loc["top1"]),
                ("localization_top3", "%.3f" % loc["top3"]),
                ("localization_coverage", "%.3f" % loc["coverage"]),
                ("localization_selective_top1", "%.3f" % loc["selective_top1"]),
                ("mean_detection_delay_s", "%.3f" % delay["mean_s"]),
                ("median_detection_delay_s", "%.3f" % delay["median_s"]),
                ("mean_detection_delay_requests", "%.3f" % delay["mean_requests"]),
                ("delay_runs_detected", str(delay["n_runs"])),
                ("delay_stride_s", "%.1f" % DELAY_STRIDE_S),
                ("inference_ms_per_window", "%.4f" % infer_ms),
                ("n_windows", str(len(rows))),
                ("n_root_cause", str(int(y.sum()))),
                ("n_unaffected", str(int(len(y) - y.sum()))),
                ("n_excluded_downstream", str(kind_counts.get("downstream", 0))),
                ("n_excluded_recovering", str(kind_counts.get("recovering", 0))),
                ("n_excluded_neutral", str(kind_counts.get("neutral", 0))),
                ("n_runs_missing_audit", str(len(dataset["missing_audit"])))]:
            writer.writerow([key, value])

    # ---- per-window dump (inspectable, not a metric) ---------------------
    dump_path = os.path.join(report_root, "windows.csv")
    if rows:
        keys = ["run_id", "pair_id", "workload_id", "cluster_config_id",
                "window_end_s", "instance_id", "role", "label_kind", "label",
                "count", "dur_mean", "dur_residual", "dur_peer_ratio",
                "has_peer", "req_ttft_mean", "req_tpot_mean", "wait_mean",
                "run_mean", "throughput_per_s"]
        with open(dump_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            writer.writeheader()
            for r in rows:
                writer.writerow(r)

    # ---- human report ----------------------------------------------------
    md_path = os.path.join(report_root, "report.md")
    with open(md_path, "w") as f:
        f.write("# Fault-detection ML report (transparent baselines)\n\n")
        f.write("Windows: %d (%d root-cause, %d unaffected), window: %.1fs, "
                "split by RUN, seed: %d.\n\n"
                % (len(rows), int(y.sum()), int(len(y) - y.sum()),
                   args.window_s, args.seed))
        f.write("Excluded from the metric: downstream %d, recovering %d, "
                "neutral %d. Runs with no audit activity: %d.\n\n"
                % (kind_counts.get("downstream", 0), kind_counts.get("recovering", 0),
                   kind_counts.get("neutral", 0), len(dataset["missing_audit"])))
        if dataset["missing_audit"]:
            f.write("Runs without scoreable audit activity: %s\n\n"
                    % ", ".join(dataset["missing_audit"]))
        f.write("| baseline | split | P | R | F1 | FPR | R 95% CI | FPR 95% CI | "
                "TP/FP/FN/TN |\n|---|---|---|---|---|---|---|---|---|\n")
        for row in table:
            f.write("| %s | %s | %.3f | %.3f | %.3f | %.3f | [%.2f, %.2f] | "
                    "[%.2f, %.2f] | %d/%d/%d/%d |\n"
                    % (row["baseline"], row["split"], row["precision"],
                       row["recall"], row["f1"], row["fpr"],
                       row["recall_ci"][0], row["recall_ci"][1],
                       row["fpr_ci"][0], row["fpr_ci"][1],
                       row["tp"], row["fp"], row["fn"], row["tn"]))
        if table and table[0]["boot_ci"]:
            f.write("\nBootstrap over runs (1000 resamples, test split, "
                    "2.5-97.5 percentile):\n\n")
            for row in table:
                if row["split"] != "test" or not row["boot_ci"]:
                    continue
                (p_lo, p_hi), (r_lo, r_hi), (f_lo, f_hi), (fp_lo, fp_hi) = row["boot_ci"]
                f.write("- %s: P [%.2f, %.2f], R [%.2f, %.2f], F1 [%.2f, %.2f], "
                        "FPR [%.2f, %.2f]\n"
                        % (row["baseline"], p_lo, p_hi, r_lo, r_hi,
                           f_lo, f_hi, fp_lo, fp_hi))
        f.write("\n## Localisation (per-instance residual ranking)\n\n")
        f.write("Scored windows: %d. Top-1 %.3f, top-3 %.3f. "
                "Abstained windows: %d (coverage %.3f), selective top-1 %.3f.\n"
                % (loc["scored"], loc["top1"], loc["top3"],
                   loc["abstained_windows"], loc["coverage"],
                   loc["selective_top1"]))
        f.write("\n## Detection delay\n\n")
        f.write("Runs with an alarm: %d. Mean %.2fs / median %.2fs after onset, "
                "%.1f requests completed in between (stride %.1fs).\n"
                % (delay["n_runs"], delay["mean_s"], delay["median_s"],
                   delay["mean_requests"], DELAY_STRIDE_S))
        f.write("\n## Overhead\n\nPredict-only cost: %.4f ms/window "
                "(fitting excluded).\n" % infer_ms)
        f.write("\nLow-severity faults may be undetectable; those negatives "
                "are reported as low recall, not tuned away.\n")
    print("wrote %s, %s and %s (%d windows)" % (csv_path, md_path, dump_path, len(rows)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
