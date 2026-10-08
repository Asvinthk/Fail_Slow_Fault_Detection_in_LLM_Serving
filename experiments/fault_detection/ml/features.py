"""Observable-only window features for fail-slow detection.

Every feature is computed from batch_telemetry.csv / request_telemetry.csv
only. A unit test asserts no manifest/audit column name is referenced
anywhere in this module's feature builder. Window aggregates are computed
offline from the two files; window length is configurable and each window
is stamped with window-end sim time.

Role per instance is resolved from the cluster config (instance order),
which is observable simulator input, not ground truth.
"""

import csv
import json
import os

import numpy as np

# Ground-truth column names that must never be reachable here.
FORBIDDEN_COLUMNS = frozenset({
    "fault_present", "fault_id", "failure_class", "successfully_applied",
    "pair_id", "expected_label", "experiment_id", "faults", "workload_id",
})

DURATION_KEYS = ("total_len", "num_prefill", "n_decode", "kv_len")


def read_csv_rows(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def instance_roles(cluster_path):
    with open(cluster_path) as f:
        cluster = json.load(f)
    roles = {}
    idx = 0
    for node in cluster.get("nodes", []):
        for inst in node.get("instances", []):
            roles[idx] = inst.get("pd_type", "colocated")
            idx += 1
    return roles


def _as_float(row, key, default=0.0):
    try:
        return float(row.get(key, default))
    except (TypeError, ValueError):
        return default


def build_windows(telemetry_dir, window_s=2.0, role="decode"):
    """Aggregate batch telemetry into fixed windows for one run dir.

    Returns a list of dicts, one per (window_end_s, instance_id) with
    observable-only statistics. Windows with no batches are skipped.
    """
    path = os.path.join(telemetry_dir, "batch_telemetry.csv")
    if not os.path.exists(path):
        return []
    rows = read_csv_rows(path)
    window_ns = int(window_s * 1e9)
    buckets = {}
    for row in rows:
        t = int(float(row.get("batch_time_ns", 0)))
        inst = int(float(row.get("instance_id", 0)))
        key = (t // window_ns, inst)
        buckets.setdefault(key, []).append(row)
    windows = []
    for (w, inst), brows in sorted(buckets.items()):
        durations = np.array([_as_float(r, "duration_ns") for r in brows])
        waits = np.array([_as_float(r, "n_waiting") for r in brows])
        runs = np.array([_as_float(r, "n_running") for r in brows])
        coms = np.array([_as_float(r, "com_cycle") for r in brows])
        kv_send = np.array([_as_float(r, "pd_kv_send_tokens") for r in brows])
        windows.append({
            "window_end_s": float((w + 1) * window_ns) / 1e9,
            "instance_id": inst,
            "role": role,
            "count": len(brows),
            "dur_mean": float(np.mean(durations)),
            "dur_p50": float(np.percentile(durations, 50)),
            "dur_p99": float(np.percentile(durations, 99)) if len(durations) > 1 else float(durations[0]),
            "dur_max": float(np.max(durations)),
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
        })
    return windows


def request_stage_stats(telemetry_dir):
    """Per-run stage-latency summary from request telemetry (observable)."""
    path = os.path.join(telemetry_dir, "request_telemetry.csv")
    if not os.path.exists(path):
        return {}
    rows = read_csv_rows(path)
    ttfts, itl_means, queues, handoffs = [], [], [], []
    for row in rows:
        ttfts.append(_as_float(row, "ttft_ns", -1))
        itl_means.append(_as_float(row, "itl_mean_ns", -1))
        first = _as_float(row, "first_scheduled_ns", -1)
        arr = _as_float(row, "arrival_ns", 0)
        if first >= 0:
            queues.append(first - arr)
        admit = _as_float(row, "decode_admit_ns", -1)
        fdd = _as_float(row, "first_decode_done_ns", -1)
        if admit >= 0 and fdd >= 0:
            handoffs.append(fdd - admit)
    ttfts = [v for v in ttfts if v >= 0]
    itl_means = [v for v in itl_means if v >= 0]

    def q(a, p):
        return float(np.percentile(a, p)) if a else -1.0
    return {
        "ttft_mean": float(np.mean(ttfts)) if ttfts else -1.0,
        "ttft_p99": q(ttfts, 99),
        "queue_mean": float(np.mean(queues)) if queues else -1.0,
        "handoff_mean": float(np.mean(handoffs)) if handoffs else -1.0,
        "itl_mean": float(np.mean(itl_means)) if itl_means else -1.0,
        "itl_p99": q(itl_means, 99),
        "n_requests": len(rows),
    }


class HealthyResidualModel:
    """Healthy-only regressions conditioned on batch shape.

    Fits on HEALTHY runs only: iteration duration vs
    (total_len, num_prefill, n_decode, kv_len); exposed-comm vs
    pd_kv_send_tokens. Residuals (observed - predicted) are the
    workload-aware signal: a slow batch has a large positive residual
    even when its absolute latency looks like a benign heavy batch.
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
        rows = [r for r in healthy_batch_rows if _as_float(r, "duration_ns") > 0]
        if len(rows) < len(DURATION_KEYS) + 1:
            return False
        X = np.column_stack(
            [np.ones(len(rows))] +
            [np.array([_as_float(r, k) for r in rows]) for k in DURATION_KEYS]
        )
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
    "com_mean", "kv_send_mean",
    "total_len_mean", "num_prefill_mean", "n_decode_mean", "kv_len_mean",
    "dur_residual", "com_residual",
    "dur_peer_ratio", "wait_peer_ratio",
]


def add_peer_and_residual_features(windows_by_run, residual_model):
    """Peer-relative ratios need >=2 same-role instances in the run.

    A fault hitting all peers symmetrically is invisible to peer
    ratios; that limitation is documented, not worked around.
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
                peer_dur = float(np.mean([p["dur_mean"] for p in peers]))
                peer_wait = float(np.mean([p["wait_mean"] for p in peers]))
                w["dur_peer_ratio"] = w["dur_mean"] / peer_dur if peer_dur > 0 else 1.0
                w["wait_peer_ratio"] = w["wait_mean"] / peer_wait if peer_wait > 0 else 1.0
            else:
                w["dur_peer_ratio"] = 1.0
                w["wait_peer_ratio"] = 1.0
    return windows_by_run


def window_to_vector(window):
    return np.array([window[c] for c in FEATURE_COLUMNS], dtype=float)
