"""Clean telemetry, build features, label, detect, score, report.

Evaluation side of the pipeline, in one file because the stages share one
contract: the SAME clock (`finish_ns`) buckets features and labels, and the
SAME dedupe rule decides what one batch is. Splitting them across modules
is what allowed the earlier instrument to bucket by schedule time while
labelling by window end, and to count a TP=2 batch twice.

Stages
------
1. load       telemetry CSVs per run; drop start-up ("phantom") reports;
              dedupe to one row per (instance, batch_id).
2. roles      resolve instance -> prefill/decode from request telemetry
              (observable), never from a single run-level argument.
3. features   per (window, instance): duration stats, queue/run counts,
              comm volume, shape, request-stage aggregates, healthy-only
              regression residuals, leave-one-out peer ratios.
4. labels     per (window, instance) from the injection AUDIT, not from
              the fault schedule: root_cause / neutral / downstream /
              recovering / unaffected, labelled `decode:1`,
              `kv-transfer-network:0`, ...
5. detect     transparent baselines only: robust threshold (median+k*MAD),
              residual threshold, EWMA, CUSUM. Logistic regression is
              opt-in and is not defensible at this sample size.
6. score      run-level (never window-level) train/test split; Wilson
              intervals for proportions; bootstrap over RUNS for P/R/F1/FPR;
              excluded kinds reported separately, never counted as FP.
7. localise   per-instance residual ranking with abstention; top-1/top-3
              against the instance-level label.
8. report     metrics.csv, windows.csv (inspectable dump), report.md.

Honesty rules baked in: every metric row carries its split and its run
count; a low-severity fault that cannot be seen shows up as low recall and
is not tuned away; downstream/recovering/neutral windows are reported, not
silently counted as false alarms.

Command
-------
  python -m experiments.fault_detection.evaluate \
      --data experiments/out/review_min --report experiments/report/review_min
"""

import argparse
import csv
import json
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from experiments.fault_detection.run_simulation import (  # noqa: E402
    MANIFEST_ONLY_COLUMNS,
    expected_label_for_fault,
    load_audit,
    load_manifest,
)

try:
    from sklearn.linear_model import LogisticRegression
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

# Ground-truth column names that must never be reachable from the feature
# builder. (run_simulation owns the canonical list; this alias keeps the
# "observable only" contract readable where features are built.)
FORBIDDEN_COLUMNS = MANIFEST_ONLY_COLUMNS

DURATION_KEYS = ("total_len", "num_prefill", "n_decode", "kv_len")

# A start-up report: every NPU emits one report with id 0 before it has any
# batch. It carries no shape and no duration.
PHANTOM_BATCH_ID = 0

# batch_telemetry.csv's `batch_id` column is the ASTRA-Sim REPORT id, which
# is the scheduler's batch id + 1 (`add_done` subtracts 1 before looking the
# batch up in batch_meta). injection_audit.jsonl records the scheduler's
# batch id. Joining the two without this offset silently shifts every label
# by one batch; the tests assert the offset.
REPORT_ID_OFFSET = 1

DELAY_STRIDE_S = 0.5
MAD_K = 4.0
EWMA_ALPHA = 0.5
CUSUM_SLACK_SIGMA = 0.5
CUSUM_THRESHOLD_SIGMA = 5.0

# Label kinds and how the metric treats them.
KIND_ROOT = "root_cause"
KIND_NEUTRAL = "neutral"
KIND_DOWNSTREAM = "downstream"
KIND_RECOVERING = "recovering"
KIND_UNAFFECTED = "unaffected"

POSITIVE_KINDS = (KIND_ROOT,)
NEGATIVE_KINDS = (KIND_UNAFFECTED,)
EXCLUDED_KINDS = (KIND_DOWNSTREAM, KIND_RECOVERING, KIND_NEUTRAL)

DEFAULT_APPLIED_SHARE = 0.5
DEFAULT_RECOVERY_S = 2.0


# ===========================================================================
# 1. load
# ===========================================================================
def read_csv_rows(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def _as_float(row, key, default=0.0):
    try:
        return float(row.get(key, default))
    except (TypeError, ValueError):
        return default


def _is_phantom(row):
    if int(_as_float(row, "batch_id", 0)) <= PHANTOM_BATCH_ID:
        return True
    return _as_float(row, "duration_ns") == 0 and _as_float(row, "total_len") == 0


def dedupe_batches(rows):
    """One row per (instance_id, batch_id), keeping the start NPU's row.

    A batch is reported once per reporting NPU: TP=2 decode emits two rows,
    and a PD prefill emits a sender row plus a receiver row in its shadow
    range. The start NPU is the smallest sys for that instance, so
    min(sys) picks it deterministically. Returns [(row, multiplicity)].
    """
    best = {}
    multiplicity = {}
    for row in rows:
        if _is_phantom(row):
            continue
        key = (int(_as_float(row, "instance_id")), int(_as_float(row, "batch_id")))
        multiplicity[key] = multiplicity.get(key, 0) + 1
        sys_id = _as_float(row, "sys", 0.0)
        if key not in best or sys_id < _as_float(best[key], "sys", 0.0):
            best[key] = row
    return [(best[k], multiplicity[k]) for k in sorted(best)]


def _window_index(row, window_ns):
    """Window index from FINISH time -- the clock the labels use too."""
    t = _as_float(row, "finish_ns", -1)
    if t < 0:
        t = _as_float(row, "batch_time_ns")
    return int(t // window_ns)


# ===========================================================================
# 2. roles
# ===========================================================================
def instance_roles(cluster_path):
    """Instance index -> pd_type, from the cluster config (simulator input)."""
    with open(cluster_path) as f:
        cluster = json.load(f)
    roles = {}
    idx = 0
    for node in cluster.get("nodes", []):
        for inst in node.get("instances", []):
            roles[idx] = inst.get("pd_type", "colocated")
            idx += 1
    return roles


def infer_instance_roles(telemetry_dir):
    """Instance index -> role, inferred from request telemetry only.

    Observable by construction: a request records which instance served its
    prefill and which served its decode. An instance that only ever appears
    as a prefill (decode) endpoint is a prefill (decode) worker; one that
    appears as both is colocated. The detector would not have the cluster
    file in production, so it must not depend on it.
    """
    path = os.path.join(telemetry_dir, "request_telemetry.csv")
    roles = {}
    if not os.path.exists(path):
        return roles
    for row in read_csv_rows(path):
        pre = _as_float(row, "prefill_instance_id", -1)
        dec = _as_float(row, "decode_instance_id", -1)
        if pre >= 0:
            roles[int(pre)] = "prefill"
        if dec >= 0:
            prev = roles.get(int(dec))
            roles[int(dec)] = "colocated" if prev == "prefill" else "decode"
    return roles


# ===========================================================================
# 3. features
# ===========================================================================
def request_stage_windows(telemetry_dir, window_s):
    """Per (window, instance) request-stage aggregates (observable).

    Decode-side rows are attributed to the request's end time, prefill-side
    rows to the prefill completion time, so a prefill window carries prefill
    service time and a decode window carries TTFT/TPOT/ITL/queue.
    """
    path = os.path.join(telemetry_dir, "request_telemetry.csv")
    if not os.path.exists(path):
        return {}, {}
    window_ns = int(window_s * 1e9)
    dec, pre = {}, {}
    for row in read_csv_rows(path):
        arr = _as_float(row, "arrival_ns", 0.0)
        end = arr + _as_float(row, "latency_ns", 0.0)
        dec_inst = int(_as_float(row, "decode_instance_id", -1))
        if dec_inst >= 0:
            key = (int(end // window_ns), dec_inst)
            d = dec.setdefault(key, {"ttft": [], "tpot": [], "itl": [], "queue": []})
            ttft = _as_float(row, "ttft_ns", -1)
            if ttft >= 0:
                d["ttft"].append(ttft)
            tpot = _as_float(row, "tpot", -1)
            if tpot >= 0:
                d["tpot"].append(tpot)
            itl = _as_float(row, "itl_p99_ns", -1)
            if itl >= 0:
                d["itl"].append(itl)
            d["queue"].append(_as_float(row, "queuing_delay_ns"))
        pre_inst = int(_as_float(row, "prefill_instance_id", -1))
        if pre_inst >= 0:
            done = _as_float(row, "prefill_done_ns", -1)
            first = _as_float(row, "first_scheduled_ns", -1)
            if done >= 0 and first >= 0:
                key = (int(done // window_ns), pre_inst)
                pre.setdefault(key, []).append(done - first)
    return dec, pre


def _merge_request_stage(window, key, dec, pre):
    window.update({
        "req_count": 0.0, "req_ttft_mean": -1.0, "req_tpot_mean": -1.0,
        "req_itl_p99_mean": -1.0, "req_queue_mean": -1.0,
        "req_prefill_svc_mean": -1.0,
    })
    d = dec.get(key)
    if d:
        window["req_count"] = float(len(d["queue"]))
        for name, values in (("req_ttft_mean", d["ttft"]),
                             ("req_tpot_mean", d["tpot"]),
                             ("req_itl_p99_mean", d["itl"]),
                             ("req_queue_mean", d["queue"])):
            if values:
                window[name] = float(np.mean(values))
    p = pre.get(key)
    if p:
        window["req_prefill_svc_mean"] = float(np.mean(p))


def build_windows(telemetry_dir, window_s=2.0, roles=None, role=None):
    """Aggregate batch telemetry into fixed windows for one run dir.

    ``roles`` maps instance_id -> role. When omitted, roles are inferred
    from request telemetry. ``role`` is a legacy fallback for single-role
    unit tests and is only used when inference yields nothing.

    Returns one dict per (window_end_s, instance_id). Windows with no
    batches are skipped.
    """
    path = os.path.join(telemetry_dir, "batch_telemetry.csv")
    if not os.path.exists(path):
        return []
    rows = read_csv_rows(path)
    if roles is None:
        roles = infer_instance_roles(telemetry_dir)
    window_ns = int(window_s * 1e9)
    buckets = {}
    for row, multiplicity in dedupe_batches(rows):
        key = (_window_index(row, window_ns), int(_as_float(row, "instance_id")))
        buckets.setdefault(key, []).append((row, multiplicity))
    dec_stage, pre_stage = request_stage_windows(telemetry_dir, window_s)
    windows = []
    for (w, inst), pairs in sorted(buckets.items()):
        brows = [p[0] for p in pairs]
        npus = np.array([float(p[1]) for p in pairs])
        durations = np.array([_as_float(r, "duration_ns") for r in brows])
        waits = np.array([_as_float(r, "n_waiting") for r in brows])
        runs = np.array([_as_float(r, "n_running") for r in brows])
        coms = np.array([_as_float(r, "com_cycle") for r in brows])
        kv_send = np.array([_as_float(r, "pd_kv_send_tokens") for r in brows])
        window = {
            "window_end_s": float((w + 1) * window_ns) / 1e9,
            "instance_id": inst,
            "role": roles.get(inst, role or "unknown"),
            "count": len(brows),
            "npus_reporting_mean": float(np.mean(npus)),
            "dur_mean": float(np.mean(durations)),
            "dur_p50": float(np.percentile(durations, 50)),
            "dur_p99": (float(np.percentile(durations, 99))
                        if len(durations) > 1 else float(durations[0])),
            "dur_max": float(np.max(durations)),
            # Batches per second (deduplicated), not NPU reports per second.
            "throughput_per_s": len(brows) / window_s,
            "wait_mean": float(np.mean(waits)),
            "wait_max": float(np.max(waits)),
            "run_mean": float(np.mean(runs)),
            "com_mean": float(np.mean(coms)),
            "kv_send_mean": float(np.mean(kv_send)),
            "total_len_mean": float(np.mean([_as_float(r, "total_len") for r in brows])),
            "num_prefill_mean": float(np.mean([_as_float(r, "num_prefill") for r in brows])),
            "n_decode_mean": float(np.mean([_as_float(r, "n_decode") for r in brows])),
            "kv_len_mean": float(np.mean([_as_float(r, "kv_len") for r in brows])),
        }
        _merge_request_stage(window, (w, inst), dec_stage, pre_stage)
        windows.append(window)
    return windows


class HealthyResidualModel:
    """Healthy-only regressions conditioned on batch shape.

    Fits on HEALTHY runs only: iteration duration vs (total_len, num_prefill,
    n_decode, kv_len); exposed-comm vs pd_kv_send_tokens. Residuals
    (observed - predicted) are the workload-aware signal: a slow batch has a
    large positive residual even when its absolute latency looks like a
    benign heavy batch.
    """

    def __init__(self):
        self.coef_ = None
        self.com_coef_ = None

    @staticmethod
    def _design(rows):
        cols = [np.ones(len(rows))]
        for k in DURATION_KEYS:
            cols.append(np.array([_as_float(r, k) for r in rows]))
        return np.column_stack(cols)

    def fit(self, healthy_batch_rows):
        rows = [r for r in healthy_batch_rows
                if _as_float(r, "duration_ns") > 0 and not _is_phantom(r)]
        if len(rows) < len(DURATION_KEYS) + 1:
            return False
        X = self._design(rows)
        y = np.array([_as_float(r, "duration_ns") for r in rows])
        self.coef_, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        Xc = np.column_stack([np.ones(len(rows)),
                              np.array([_as_float(r, "pd_kv_send_tokens") for r in rows])])
        yc = np.array([_as_float(r, "com_cycle") for r in rows])
        self.com_coef_, _, _, _ = np.linalg.lstsq(Xc, yc, rcond=None)
        return True

    def residual(self, window):
        if self.coef_ is None:
            return 0.0
        x = np.array([1.0, window["total_len_mean"],
                      window["num_prefill_mean"], window["n_decode_mean"],
                      window["kv_len_mean"]])
        return window["dur_mean"] - float(x @ self.coef_)

    def com_residual(self, window):
        if self.com_coef_ is None:
            return 0.0
        x = np.array([1.0, window["kv_send_mean"]])
        return window["com_mean"] - float(x @ self.com_coef_)


FEATURE_COLUMNS = [
    "dur_mean", "dur_p50", "dur_p99", "dur_max",
    "throughput_per_s", "wait_mean", "wait_max", "run_mean",
    "com_mean", "kv_send_mean", "npus_reporting_mean",
    "total_len_mean", "num_prefill_mean", "n_decode_mean", "kv_len_mean",
    "dur_residual", "com_residual",
    "dur_peer_ratio", "wait_peer_ratio", "has_peer",
    "req_count", "req_ttft_mean", "req_tpot_mean", "req_itl_p99_mean",
    "req_queue_mean", "req_prefill_svc_mean",
]


def add_peer_and_residual_features(windows_by_run, residual_model):
    """Peer-relative ratios over same-role peers, excluding self.

    The peer statistic is a leave-one-out MEDIAN: a mean that includes the
    suspect instance dilutes its own deviation, and with two peers one bad
    worker shifts the reference by half its excess. A lone instance has no
    peer, so has_peer=0 and the ratio is 1.0 by convention -- callers must
    gate on has_peer rather than trust the ratio.

    A fault hitting all peers symmetrically is invisible to peer ratios;
    that limitation is documented, not worked around.
    """
    for windows in windows_by_run:
        by_end = {}
        for w in windows:
            by_end.setdefault(w["window_end_s"], []).append(w)
        for w in windows:
            w["dur_residual"] = residual_model.residual(w)
            w["com_residual"] = residual_model.com_residual(w)
            peers = [p for p in by_end[w["window_end_s"]]
                     if p["instance_id"] != w["instance_id"]
                     and p["role"] == w["role"]]
            if peers:
                peer_dur = float(np.median([p["dur_mean"] for p in peers]))
                peer_wait = float(np.median([p["wait_mean"] for p in peers]))
                w["dur_peer_ratio"] = w["dur_mean"] / peer_dur if peer_dur > 0 else 1.0
                w["wait_peer_ratio"] = w["wait_mean"] / peer_wait if peer_wait > 0 else 1.0
                w["has_peer"] = 1.0
            else:
                w["dur_peer_ratio"] = 1.0
                w["wait_peer_ratio"] = 1.0
                w["has_peer"] = 0.0
    return windows_by_run


def window_to_vector(window):
    return np.array([window[c] for c in FEATURE_COLUMNS], dtype=float)


# ===========================================================================
# 4. labels (from the audit)
# ===========================================================================
def _window_key(window_end_s, instance_id):
    return (round(float(window_end_s), 6), int(instance_id))


def _fault_targets(entry):
    """fault_id -> (component, instance_id) for one manifest entry."""
    targets = {}
    for fault in entry.get("faults", []):
        component = expected_label_for_fault(fault)
        instance = fault.get("target", {}).get("instance_id", -1)
        targets[fault.get("fault_id")] = (component, int(instance))
    return targets


def label_windows(telemetry_dir, entry, window_s,
                  applied_share=DEFAULT_APPLIED_SHARE,
                  recovery_s=DEFAULT_RECOVERY_S):
    """Return {(window_end_s, instance_id): label_dict} for one run.

    A window is a root-cause positive when the audit shows the mechanism
    applied to THAT instance in THAT window. Healthy runs yield only
    `unaffected`.
    """
    window_ns = int(window_s * 1e9)
    labels = {}

    # Batch inventory per (window, instance), from observable telemetry.
    path = os.path.join(telemetry_dir, "batch_telemetry.csv")
    rows = read_csv_rows(path) if os.path.exists(path) else []
    batch_window = {}
    counts = {}
    for row, _multiplicity in dedupe_batches(rows):
        inst = int(_as_float(row, "instance_id"))
        widx = _window_index(row, window_ns)
        bid = int(_as_float(row, "batch_id"))
        batch_window[(inst, bid)] = widx
        counts[(widx, inst)] = counts.get((widx, inst), 0) + 1

    # Audit -> per-(window, instance) applied / neutral batch counts.
    applied = {}
    neutral = {}
    for record in load_audit(telemetry_dir):
        inst = record.get("instance")
        bid = record.get("batch_id")
        if inst is None:
            continue
        if bid is None:
            # Event-keyed records: a kv_handoff_delay hold, or a
            # worker_stall window summary. They carry no batch, so they are
            # attributed to the window containing the event time. The hold
            # is a transfer-link event owned by the prefill instance, so the
            # label becomes kv-transfer-network:<prefill>.
            if record.get("applied") and record.get("time_ns") is not None:
                widx = int(float(record["time_ns"]) // window_ns)
                applied[(widx, int(inst))] = applied.get((widx, int(inst)), 0) + 1
            continue
        key = batch_window.get((int(inst), int(bid) + REPORT_ID_OFFSET))
        if key is None:
            continue
        if record.get("applied"):
            applied[(key, int(inst))] = applied.get((key, int(inst)), 0) + 1
        else:
            neutral[(key, int(inst))] = neutral.get((key, int(inst)), 0) + 1

    targets = _fault_targets(entry)
    component_of_instance = {}
    for component, instance in targets.values():
        component_of_instance.setdefault(instance, component)

    # Windows in which SOME instance was affected: used for downstream.
    active_windows = {w for (w, _inst) in applied}
    active_windows |= {w for (w, _inst) in neutral}

    # Recovery horizon from the schedule (the audit has no "end" record).
    recovery_windows = set()
    for fault in entry.get("faults", []):
        end_s = fault.get("schedule", {}).get("end_s")
        if end_s is None:
            continue
        lo = int(end_s * 1e9) // window_ns
        hi = int((end_s + recovery_s) * 1e9) // window_ns
        recovery_windows.update(range(lo, hi + 1))

    for (widx, inst), batches in counts.items():
        window_end_s = float((widx + 1) * window_ns) / 1e9
        a = applied.get((widx, inst), 0)
        z = neutral.get((widx, inst), 0)
        component = component_of_instance.get(inst)
        if a and batches:
            share = min(1.0, a / float(batches))
            kind = KIND_ROOT
            label = "%s:%d" % (component or "unknown", inst)
        elif z:
            share = 0.0
            kind = KIND_NEUTRAL
            label = "%s:%d" % (component or "unknown", inst)
        elif widx in active_windows:
            share = 0.0
            kind = KIND_DOWNSTREAM
            label = "downstream:%d" % inst
        elif widx in recovery_windows:
            share = 0.0
            kind = KIND_RECOVERING
            label = "recovering:%d" % inst
        else:
            share = 0.0
            kind = KIND_UNAFFECTED
            label = "unaffected"
        labels[_window_key(window_end_s, inst)] = {
            "kind": kind,
            "component": (component if kind in (KIND_ROOT, KIND_NEUTRAL)
                          else ("downstream" if kind == KIND_DOWNSTREAM
                                else ("recovering" if kind == KIND_RECOVERING
                                      else "healthy"))),
            "instance_id": inst,
            "label": label,
            "applied_share": share,
            "partial": share < applied_share,
            "batches": batches,
            "window_end_s": window_end_s,
        }
    return labels


def attach_labels(windows, labels):
    """Attach each window's label dict; unlabelled windows are marked."""
    out = []
    for window in windows:
        key = _window_key(window["window_end_s"], window["instance_id"])
        lab = labels.get(key) or {
            "kind": "unlabelled", "component": "unknown",
            "instance_id": window["instance_id"], "label": "unlabelled",
            "applied_share": 0.0, "partial": False, "batches": 0,
            "window_end_s": window["window_end_s"],
        }
        merged = dict(window)
        merged.update({
            "label_kind": lab["kind"],
            "label": lab["label"],
            "label_component": lab["component"],
            "label_instance_id": lab["instance_id"],
            "label_applied_share": lab["applied_share"],
            "label_partial": lab["partial"],
        })
        out.append(merged)
    return out


# ===========================================================================
# 5. statistics
# ===========================================================================
def wilson(k, n, z=1.96):
    """Wilson score interval for a proportion (k of n)."""
    if n <= 0:
        return 0.0, 0.0
    p = k / float(n)
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2.0 * n)) / denom
    half = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def robust_sigma(values):
    """1.4826 * MAD, with a floor so a deterministic run is not zero-width."""
    values = [v for v in values if v is not None and np.isfinite(v)]
    if not values:
        return 0.0, 0.0
    arr = np.array(values, dtype=float)
    med = float(np.median(arr))
    mad = float(np.median(np.abs(arr - med)))
    sigma = 1.4826 * mad
    if sigma <= 0:
        sigma = float(np.std(arr))
    return med, sigma


def mad_threshold(values, k=MAD_K):
    """Robust upper threshold: median + k * 1.4826 * MAD.

    A p99 over a handful of healthy windows IS the maximum, so a single
    benign spike sets the threshold for everything. MAD is break-down
    robust. When the healthy set has zero spread (no timing jitter), the
    floor keeps the threshold just above the median instead of collapsing
    onto it.
    """
    med, sigma = robust_sigma(values)
    if sigma <= 0:
        sigma = 0.0
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


def bootstrap_over_runs(pred_by_run, true_by_run, n_boot=1000, seed=42):
    """Percentile CI for P/R/F1/FPR, resampling RUNS (not windows)."""
    unique = sorted(pred_by_run)
    if len(unique) < 2:
        return None
    rng = np.random.RandomState(seed)
    stats = []
    for _ in range(n_boot):
        pick = rng.choice(len(unique), size=len(unique), replace=True)
        tp = fp = fn = tn = 0
        for idx in pick:
            run = unique[idx]
            a, b, c, d = prf_counts(true_by_run[run], pred_by_run[run])
            tp += a
            fp += b
            fn += c
            tn += d
        stats.append(prf_from_counts(tp, fp, fn, tn))
    arr = np.array(stats)
    return [tuple(np.percentile(arr[:, i], [2.5, 97.5])) for i in range(4)]


# ===========================================================================
# 6. dataset
# ===========================================================================
def _telemetry_dir(data_root, entry):
    """Resolve a run's telemetry directory from its manifest entry.

    A pair with several fault variants writes `faulty0_telemetry`,
    `faulty1_telemetry`, ... rather than one `faulty_telemetry`, so the
    directory cannot be derived from `fault_present` alone. Preference
    order:

    1. `telemetry_dir`, recorded by inject_faults (current bundles);
    2. the `run_id` tail, which is exactly the directory stem
       (`<pair>_faulty3` -> `faulty3_telemetry`); bundles written before
       the field existed still resolve to their own twin, not to twin 0;
    3. the healthy/faulty default, for a single-variant pair.
    """
    pair_dir = os.path.join(data_root, entry["pair_id"])
    suffix = "faulty" if entry.get("fault_present") else "healthy"
    names = []
    if entry.get("telemetry_dir"):
        names.append(entry["telemetry_dir"])
    tail = str(entry.get("run_id", "")).rsplit("_", 1)[-1]
    if tail.startswith("faulty") or tail == "healthy":
        names.append("%s_telemetry" % tail)
    names.append("%s_telemetry" % suffix)
    for name in names:
        candidate = os.path.join(pair_dir, name)
        if os.path.exists(os.path.join(candidate, "batch_telemetry.csv")):
            return candidate
    return os.path.join(pair_dir, names[0])


def collect_dataset(data_root, window_s, applied_share, recovery_s):
    """Load every run, build windows, attach audit-derived labels."""
    manifest_path = os.path.join(data_root, "manifest.jsonl")
    if not os.path.exists(manifest_path):
        print("no manifest.jsonl under %s; run inject_faults first" % data_root)
        return None
    manifest = [e for e in load_manifest(manifest_path) if e.get("status") == "ok"]

    # Timing-neutral mechanisms. A fault can change the trace (the audit
    # says applied=True) and still carry no latency signal. In a
    # deterministic simulator that is provable from the total clocks:
    #
    #   * two or more severities of the same mechanism on the same target
    #     producing bit-identical clocks => the severity knob does nothing
    #     (kv_transfer_bandwidth at 0.8/0.67/0.5 is exactly this);
    #   * a single severity that reproduces the healthy twin's clocks
    #     => zero effect.
    #
    # Scoring such windows as positives would count a guaranteed miss
    # against every detector. They become `neutral` (excluded) and the
    # family is named in the report.
    healthy_clocks = {e["pair_id"]: e.get("total_clocks_ns")
                      for e in manifest if not e.get("fault_present")}
    clocks_by_family = {}
    for e in manifest:
        if e.get("fault_present") and e.get("faults"):
            key = (e["pair_id"], e["faults"][0]["fault_id"])
            clocks_by_family.setdefault(key, []).append(e.get("total_clocks_ns"))
    neutral_families = sorted(
        key for key, clocks in clocks_by_family.items()
        # Unknown clocks prove nothing: never call a fault neutral on a
        # missing measurement.
        if None not in clocks
        and ((len(clocks) >= 2 and len(set(clocks)) == 1)
             or (len(clocks) == 1 and clocks[0] == healthy_clocks.get(key[0]))))

    windows_by_run = []
    rows = []
    missing_audit = []
    skipped = []
    for entry in manifest:
        telemetry_dir = _telemetry_dir(data_root, entry)
        if not os.path.exists(os.path.join(telemetry_dir, "batch_telemetry.csv")):
            # Never skip silently: a directory-name mismatch here once
            # dropped every faulty run and yielded an all-unaffected
            # dataset that looked like "the detector found nothing".
            skipped.append(entry["run_id"])
            continue
        roles = infer_instance_roles(telemetry_dir)
        windows = build_windows(telemetry_dir, window_s=window_s, roles=roles)
        labels = label_windows(telemetry_dir, entry, window_s,
                               applied_share=applied_share,
                               recovery_s=recovery_s)
        family = (entry["pair_id"], entry["faults"][0]["fault_id"]
                  if entry.get("faults") else None)
        if family in neutral_families:
            for lab in labels.values():
                if lab["kind"] == KIND_ROOT:
                    lab["kind"] = KIND_NEUTRAL
                    lab["label"] = "neutral:%d" % lab["instance_id"]
                    lab["note"] = ("timing-neutral mechanism: no clock "
                                   "response to severity")
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
            w["regime"] = entry.get("regime", {})
        windows_by_run.append(windows)
        rows.extend(windows)

    # Healthy-only residual model, fitted on raw healthy batches.
    healthy_rows = []
    for entry in manifest:
        if entry.get("fault_present"):
            continue
        path = os.path.join(_telemetry_dir(data_root, entry),
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
        "skipped_runs": skipped,
        "timing_neutral_faults": ["%s/%s" % key for key in neutral_families],
    }


def eligible_rows(rows):
    """Rows that enter the detection metric (root_cause vs unaffected)."""
    return [r for r in rows
            if r["label_kind"] in POSITIVE_KINDS or r["label_kind"] in NEGATIVE_KINDS]


# ===========================================================================
# 7. detectors
# ===========================================================================
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


def cusum_predict(rows, key, role_params):
    """Tabular CUSUM on the residual, per (run, instance).

    s_t = max(0, s_{t-1} + (x_t - k)); alarm when s_t > h. k and h come
    from the healthy residual spread of that role, so a role whose healthy
    residual is naturally wide is not alarm-prone. EWMA reacts to a single
    large window; CUSUM accumulates small persistent shifts, which is what
    a 1.25x fault produces.
    """
    by_key = {}
    for idx, r in enumerate(rows):
        by_key.setdefault((r["run_id"], r["instance_id"]), []).append((idx, r))
    pred = np.zeros(len(rows), dtype=int)
    for (run, inst), items in by_key.items():
        items.sort(key=lambda x: x[1]["window_end_s"])
        k, h = role_params.get(items[0][1]["role"], (0.0, float("inf")))
        s = 0.0
        for idx, r in items:
            s = max(0.0, s + (r[key] - k))
            pred[idx] = 1 if s > h else 0
    return pred


def predict_series(rows, series, thresholds, key):
    pred = np.zeros(len(rows), dtype=int)
    for i, r in enumerate(rows):
        value = series.get(((r["run_id"], r["instance_id"]), r["window_end_s"]))
        if value is None:
            continue
        pred[i] = 1 if value > thresholds.get(r["role"], float("inf")) else 0
    return pred


def cusum_params(train_rows, key):
    """(k, h) per role from healthy residual spread."""
    params = {}
    for role in sorted({r["role"] for r in train_rows}):
        values = [r[key] for r in train_rows
                  if r["role"] == role and r["label_kind"] in NEGATIVE_KINDS]
        _med, sigma = robust_sigma(values)
        sigma = max(sigma, 1.0)
        params[role] = (CUSUM_SLACK_SIGMA * sigma,
                        CUSUM_THRESHOLD_SIGMA * sigma)
    return params


# ===========================================================================
# 8. localisation
# ===========================================================================
def localisation(all_rows, thresholds, key="dur_residual"):
    """Rank instances by residual inside each window; score top-1/top-3.

    Only root_cause windows are scored. The candidate set is EVERY instance
    present in that window (including healthy ones and downstream victims),
    which is the realistic task: the localiser does not know how many
    faults exist. Abstention is allowed when no instance exceeds its
    role's healthy-calibrated threshold.
    """
    by_window = {}
    for r in all_rows:
        by_window.setdefault((r["run_id"], r["window_end_s"]), []).append(r)
    top1 = top3 = scored = abstained = sel_correct = sel_total = 0
    n_windows = 0
    for (_run, _end), items in by_window.items():
        positives = [r for r in items if r["label_kind"] in POSITIVE_KINDS]
        if not positives:
            continue
        n_windows += 1
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
        "windows": n_windows,
        "scored": scored,
        "top1": top1 / float(scored) if scored else 0.0,
        "top3": top3 / float(scored) if scored else 0.0,
        "top1_ci": wilson(top1, scored),
        "abstained_windows": abstained,
        "coverage": 1.0 - abstained / float(n_windows) if n_windows else 0.0,
        "selective_top1": sel_correct / float(sel_total) if sel_total else 0.0,
    }


# ===========================================================================
# 9. detection delay (0.5 s sliding stride)
# ===========================================================================
def batch_residuals(telemetry_dir, model):
    """Per-batch residual using the healthy shape model (observable only)."""
    path = os.path.join(telemetry_dir, "batch_telemetry.csv")
    if not os.path.exists(path):
        return []
    out = []
    for row, _mult in dedupe_batches(read_csv_rows(path)):
        if model.coef_ is None:
            continue
        x = np.array([1.0, _as_float(row, "total_len"),
                      _as_float(row, "num_prefill"), _as_float(row, "n_decode"),
                      _as_float(row, "kv_len")])
        out.append({
            "finish_ns": _as_float(row, "finish_ns"),
            "instance_id": int(_as_float(row, "instance_id")),
            "residual": _as_float(row, "duration_ns") - float(x @ model.coef_),
        })
    out.sort(key=lambda b: b["finish_ns"])
    return out


def detection_delay(dataset, data_root, thresholds):
    """First EWMA crossing after onset, at 0.5 s stride.

    Delay is reported in simulated seconds AND in requests completed
    between onset and the alarm, because a window-granular delay is an
    artefact of the window length.
    """
    model = dataset["model"]
    results = []
    limit_thresh = min(thresholds.values()) if thresholds else float("inf")
    for run_id, entry in dataset["manifest"].items():
        if not entry.get("fault_present"):
            continue
        starts = [f.get("schedule", {}).get("start_s") for f in entry.get("faults", [])]
        starts = [s for s in starts if s is not None]
        if not starts:
            continue
        onset = min(starts)
        run_dir = os.path.join(data_root, entry["pair_id"])
        telemetry_dir = os.path.join(run_dir, "faulty_telemetry")
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
                ewma[b["instance_id"]] = (b["residual"] if prev is None
                                          else EWMA_ALPHA * b["residual"]
                                          + (1.0 - EWMA_ALPHA) * prev)
                idx += 1
            if t >= onset and alarm_s is None:
                for value in ewma.values():
                    if value > limit_thresh:
                        alarm_s = t
                        alarm_requests = sum(1 for e in ends
                                             if onset * 1e9 < e <= limit)
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


# ===========================================================================
# 10. regime summary
# ===========================================================================
def regime_table(rows):
    """Group runs by (cluster, workload) and summarise their regime.

    Every reported number must carry the load it was measured at, so the
    report prints this table above the metrics.
    """
    groups = {}
    for r in rows:
        key = (r["cluster_config_id"], r["workload_id"])
        groups.setdefault(key, {"runs": set(), "u_decode": [], "u_prefill": [],
                                "lambda": [], "tpot": []})
        reg = r.get("regime") or {}
        g = groups[key]
        g["runs"].add(r["run_id"])
        if reg.get("u_decode", -1) >= 0:
            g["u_decode"].append(reg["u_decode"])
            g["u_prefill"].append(reg.get("u_prefill", -1))
            g["lambda"].append(reg.get("lambda_sps", -1))
            g["tpot"].append(reg.get("tpot_p50_ms", -1))
    out = []
    for (cluster, workload), g in sorted(groups.items()):
        med = (lambda a: float(np.median(a)) if a else -1.0)
        out.append({
            "cluster": cluster, "workload": workload,
            "runs": len(g["runs"]),
            "u_decode": med(g["u_decode"]), "u_prefill": med(g["u_prefill"]),
            "lambda": med(g["lambda"]), "tpot_ms": med(g["tpot"]),
        })
    return out


# ===========================================================================
# 11. main
# ===========================================================================
def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--window-s", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--applied-share", type=float, default=DEFAULT_APPLIED_SHARE,
                        help="share of an instance's batches that must carry an applied "
                             "audit record for the window to be a root-cause positive")
    parser.add_argument("--recovery-s", type=float, default=DEFAULT_RECOVERY_S,
                        help="windows this long after a fault ends are 'recovering' (excluded)")
    parser.add_argument("--with-logistic", action="store_true",
                        help="also report logistic regression (not defensible at small n)")
    parser.add_argument("--no-bootstrap", action="store_true")
    args = parser.parse_args(argv)

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    data_root = args.data if os.path.isabs(args.data) else os.path.join(repo_root, args.data)
    report_root = args.report if os.path.isabs(args.report) else os.path.join(repo_root, args.report)
    os.makedirs(report_root, exist_ok=True)

    dataset = collect_dataset(data_root, args.window_s, args.applied_share,
                              args.recovery_s)
    if dataset is None:
        return 2
    all_rows = dataset["rows"]
    if not all_rows:
        print("no windows found under %s; run inject_faults first" % data_root)
        return 2

    rows = eligible_rows(all_rows) if any(
        r["label_kind"] in POSITIVE_KINDS for r in all_rows) else all_rows
    if not rows:
        print("no scoreable windows under %s (no audit activity?)" % data_root)
        return 2

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
        # Tiny dataset: split runs 2/3 - 1/3 deterministically.
        train_pairs = set(pairs[: max(1, len(pairs) * 2 // 3)])
        test_pairs = set(pairs) - train_pairs
    is_train = np.array([r["pair_id"] in train_pairs for r in rows])
    is_test = ~is_train

    # ---- thresholds calibrated on HEALTHY TRAIN windows only ------------
    train_rows = [r for r, tr in zip(rows, is_train) if tr]
    thr_dur = role_thresholds(train_rows, "dur_mean")
    thr_res = role_thresholds(train_rows, "dur_residual")
    cusum = cusum_params(train_rows, "dur_residual")

    # ---- detectors -------------------------------------------------------
    dur = np.array([r["dur_mean"] for r in rows])
    res = np.array([r["dur_residual"] for r in rows])
    roles = [r["role"] for r in rows]

    pred_a = np.array([1 if dur[i] > thr_dur.get(roles[i], float("inf")) else 0
                       for i in range(len(rows))])
    pred_b = np.array([1 if res[i] > thr_res.get(roles[i], float("inf")) else 0
                       for i in range(len(rows))])

    # EWMA / CUSUM on the residual, predict-only timing.
    t0 = time.time()
    series = ewma_series(rows, "dur_residual")
    pred_c = predict_series(rows, series, thr_res, "dur_residual")
    pred_d = cusum_predict(rows, "dur_residual", cusum)
    infer_ms = (time.time() - t0) / max(1, len(rows)) * 1000.0

    pred_e = None
    if args.with_logistic and HAS_SKLEARN:
        y_all = np.array([1 if r["label_kind"] in POSITIVE_KINDS else 0 for r in rows])
        X = np.array([window_to_vector(r) for r in rows])
        if len(set(y_all[is_train])) == 2:
            clf = LogisticRegression(max_iter=1000, random_state=args.seed)
            clf.fit(X[is_train], y_all[is_train])
            pred_e = clf.predict(X)

    y = np.array([1 if r["label_kind"] in POSITIVE_KINDS else 0 for r in rows])

    baselines = [("fixed_threshold_robust", pred_a),
                 ("residual_robust", pred_b),
                 ("ewma_residual", pred_c),
                 ("cusum_residual", pred_d)]
    if pred_e is not None:
        baselines.append(("logistic", pred_e))

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
                pred_by_run, true_by_run = {}, {}
                for r, p, t, m in zip(rows, pred, y, mask):
                    if not m:
                        continue
                    pred_by_run.setdefault(r["pair_id"], []).append(p)
                    true_by_run.setdefault(r["pair_id"], []).append(t)
                pred_by_run = {k: np.array(v) for k, v in pred_by_run.items()}
                true_by_run = {k: np.array(v) for k, v in true_by_run.items()}
                ci = bootstrap_over_runs(pred_by_run, true_by_run, seed=args.seed)
            table.append({
                "baseline": name, "split": split,
                "precision": precision, "recall": recall, "f1": f1, "fpr": fpr,
                "tp": tp, "fp": fp, "fn": fn, "tn": tn,
                "n_runs": len({r["pair_id"] for r, m in zip(rows, mask) if m}),
                "recall_ci": (rec_lo, rec_hi), "fpr_ci": (fpr_lo, fpr_hi),
                "boot_ci": ci,
            })

    # ---- excluded-kind accounting ----------------------------------------
    kind_counts = {}
    for r in all_rows:
        kind_counts[r["label_kind"]] = kind_counts.get(r["label_kind"], 0) + 1

    loc = localisation(all_rows, thr_res)
    delay = detection_delay(dataset, data_root, thr_res)
    regimes = regime_table(all_rows)

    # ---- write metrics.csv ------------------------------------------------
    csv_path = os.path.join(report_root, "metrics.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["baseline", "split", "precision", "recall", "f1",
                         "fpr_healthy", "tp", "fp", "fn", "tn", "n_runs",
                         "recall_ci_lo", "recall_ci_hi",
                         "fpr_ci_lo", "fpr_ci_hi"])
        for row in table:
            writer.writerow([
                row["baseline"], row["split"],
                "%.3f" % row["precision"], "%.3f" % row["recall"],
                "%.3f" % row["f1"], "%.3f" % row["fpr"],
                row["tp"], row["fp"], row["fn"], row["tn"], row["n_runs"],
                "%.3f" % row["recall_ci"][0], "%.3f" % row["recall_ci"][1],
                "%.3f" % row["fpr_ci"][0], "%.3f" % row["fpr_ci"][1]])
        writer.writerow([])
        for key, value in [
                ("localization_top1", "%.3f" % loc["top1"]),
                ("localization_top1_ci_lo", "%.3f" % loc["top1_ci"][0]),
                ("localization_top1_ci_hi", "%.3f" % loc["top1_ci"][1]),
                ("localization_top3", "%.3f" % loc["top3"]),
                ("localization_coverage", "%.3f" % loc["coverage"]),
                ("localization_selective_top1", "%.3f" % loc["selective_top1"]),
                ("localization_windows", str(loc["windows"])),
                ("mean_detection_delay_s", "%.3f" % delay["mean_s"]),
                ("median_detection_delay_s", "%.3f" % delay["median_s"]),
                ("mean_detection_delay_requests", "%.3f" % delay["mean_requests"]),
                ("delay_runs_detected", str(delay["n_runs"])),
                ("delay_stride_s", "%.1f" % DELAY_STRIDE_S),
                ("inference_ms_per_window", "%.4f" % infer_ms),
                ("n_windows", str(len(rows))),
                ("n_root_cause", str(int(y.sum()))),
                ("n_unaffected", str(int(len(y) - y.sum()))),
                ("n_excluded_downstream", str(kind_counts.get(KIND_DOWNSTREAM, 0))),
                ("n_excluded_recovering", str(kind_counts.get(KIND_RECOVERING, 0))),
                ("n_excluded_neutral", str(kind_counts.get(KIND_NEUTRAL, 0))),
                ("n_runs_missing_audit", str(len(dataset["missing_audit"]))),
                ("n_runs_missing_telemetry",
                 str(len(dataset["skipped_runs"]))),
                ("n_timing_neutral_families",
                 str(len(dataset["timing_neutral_faults"])))]:
            writer.writerow([key, value])

    # ---- per-window dump (inspectable, not a metric) ----------------------
    dump_path = os.path.join(report_root, "windows.csv")
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
        f.write("# Fault-detection report (transparent baselines)\n\n")
        f.write("Synthetic faults only: these numbers validate the detection "
                "machinery, not real-hardware detection.\n\n")
        f.write("Windows: %d (%d root-cause, %d unaffected), window %.1fs, "
                "split by RUN, seed %d.\n\n"
                % (len(rows), int(y.sum()), int(len(y) - y.sum()),
                   args.window_s, args.seed))
        f.write("Excluded from the metric: downstream %d, recovering %d, "
                "neutral %d. Runs with no audit activity: %d.\n\n"
                % (kind_counts.get(KIND_DOWNSTREAM, 0),
                   kind_counts.get(KIND_RECOVERING, 0),
                   kind_counts.get(KIND_NEUTRAL, 0),
                   len(dataset["missing_audit"])))
        if dataset["missing_audit"]:
            f.write("Runs without scoreable audit activity: %s\n\n"
                    % ", ".join(dataset["missing_audit"]))
        if dataset["skipped_runs"]:
            f.write("Runs skipped because their telemetry directory is "
                    "missing (not scored, not silently treated as healthy): "
                    "%s\n\n" % ", ".join(dataset["skipped_runs"]))
        if dataset["timing_neutral_faults"]:
            f.write("Timing-neutral mechanisms (no latency signal: either "
                    "several severities gave bit-identical total clocks, or a "
                    "single severity reproduced the healthy twin's clocks). "
                    "Their windows are excluded as `neutral`, not scored as "
                    "detection targets: %s\n\n"
                    % ", ".join(dataset["timing_neutral_faults"]))

        f.write("## Regime (every number above is measured at this load)\n\n")
        f.write("| cluster | workload | runs | U_decode | U_prefill | lambda/s | TPOT p50 (ms) |\n")
        f.write("|---|---|---|---|---|---|---|\n")
        for r in regimes:
            f.write("| %s | %s | %d | %.3f | %.3f | %.3f | %.2f |\n"
                    % (r["cluster"], r["workload"], r["runs"], r["u_decode"],
                       r["u_prefill"], r["lambda"], r["tpot_ms"]))

        f.write("\n## Baselines\n\n")
        f.write("| baseline | split | runs | P | R | F1 | FPR | R 95% CI | "
                "FPR 95% CI | TP/FP/FN/TN |\n|---|---|---|---|---|---|---|---|---|---|\n")
        for row in table:
            f.write("| %s | %s | %d | %.3f | %.3f | %.3f | %.3f | [%.2f, %.2f] | "
                    "[%.2f, %.2f] | %d/%d/%d/%d |\n"
                    % (row["baseline"], row["split"], row["n_runs"],
                       row["precision"], row["recall"], row["f1"], row["fpr"],
                       row["recall_ci"][0], row["recall_ci"][1],
                       row["fpr_ci"][0], row["fpr_ci"][1],
                       row["tp"], row["fp"], row["fn"], row["tn"]))
        boot_rows = [row for row in table
                     if row["split"] == "test" and row["boot_ci"]]
        if boot_rows:
            f.write("\nBootstrap over runs (1000 resamples, 2.5-97.5 percentile):\n\n")
            for row in boot_rows:
                (p_lo, p_hi), (r_lo, r_hi), (f_lo, f_hi), (fp_lo, fp_hi) = row["boot_ci"]
                f.write("- %s: P [%.2f, %.2f], R [%.2f, %.2f], F1 [%.2f, %.2f], "
                        "FPR [%.2f, %.2f]\n"
                        % (row["baseline"], p_lo, p_hi, r_lo, r_hi,
                           f_lo, f_hi, fp_lo, fp_hi))
        f.write("\n## Localisation (per-instance residual ranking)\n\n")
        f.write("Root-cause windows: %d. Top-1 %.3f (95%% CI [%.2f, %.2f]), "
                "top-3 %.3f. Abstained windows: %d (coverage %.3f), "
                "selective top-1 %.3f.\n"
                % (loc["windows"], loc["top1"], loc["top1_ci"][0],
                   loc["top1_ci"][1], loc["top3"], loc["abstained_windows"],
                   loc["coverage"], loc["selective_top1"]))
        f.write("\n## Detection delay\n\n")
        f.write("Runs with an alarm: %d. Mean %.2fs / median %.2fs after onset, "
                "%.1f requests completed in between (stride %.1fs).\n"
                % (delay["n_runs"], delay["mean_s"], delay["median_s"],
                   delay["mean_requests"], DELAY_STRIDE_S))
        f.write("\n## Overhead\n\nPredict-only cost: %.4f ms/window "
                "(fitting excluded).\n" % infer_ms)
        f.write("\nLow-severity faults may be undetectable; those negatives are "
                "reported as low recall, not tuned away.\n")
    print("wrote %s, %s and %s (%d windows)"
          % (csv_path, md_path, dump_path, len(rows)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
