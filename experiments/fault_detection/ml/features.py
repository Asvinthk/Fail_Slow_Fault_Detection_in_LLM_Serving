"""Observable-only window features for fail-slow detection.

Every feature is computed from batch_telemetry.csv / request_telemetry.csv
only. A unit test asserts no manifest/audit column name is referenced
anywhere in this module's feature builder. Window aggregates are computed
offline from the two files; window length is configurable.

Fixes applied after external review (see report.md section 16):
  * role is resolved PER INSTANCE (config or observable inference), never
    stamped from a single argument (F2);
  * start-up reports (batch_id 0 / zero-shape rows) are dropped (F7);
  * rows are deduplicated per (instance_id, batch_id) so a TP=2 or a
    P/D sender+receiver pair counts as ONE batch, with npus_reporting
    recorded instead of silently inflating throughput (F8);
  * bucketing uses finish_ns, the same clock the labels use (N1);
  * peer ratios are leave-one-out medians and carry a has_peer flag, so a
    lone decode instance no longer yields a meaningless 1.0 (N3);
  * request-stage aggregates (TTFT/TPOT/ITL/queue/prefill service) are
    merged per (window, instance) instead of being computed and unused
    (F9);
  * run_id is never a feature, and no ground-truth column is readable
    here (N6).
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

# A start-up report: every NPU emits one report with id 0 before it has
# any batch. It carries no shape and no duration.
PHANTOM_BATCH_ID = 0


def read_csv_rows(path):
    with open(path) as f:
        return list(csv.DictReader(f))


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

    Observable by construction: a request records which instance served
    its prefill and which served its decode. An instance that only ever
    appears as a prefill (decode) endpoint is a prefill (decode) worker;
    one that appears as both is colocated. This avoids depending on the
    cluster file, which the detector would not have in production.
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


def _as_float(row, key, default=0.0):
    try:
        return float(row.get(key, default))
    except (TypeError, ValueError):
        return default


def _is_phantom(row):
    if int(_as_float(row, "batch_id", 0)) <= PHANTOM_BATCH_ID:
        return True
    return _as_float(row, "duration_ns") == 0 and _as_float(row, "total_len") == 0


def _dedupe_batches(rows):
    """One row per (instance_id, batch_id), keeping the start NPU's row.

    A batch is reported once per reporting NPU: TP=2 decode emits two
    rows, and a PD prefill emits a sender row plus a receiver row in its
    shadow range. The start NPU is the smallest sys for that instance
    (inst2npu_mapping), so min(sys) picks it deterministically.
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
    t = _as_float(row, "finish_ns", -1)
    if t < 0:
        t = _as_float(row, "batch_time_ns")
    return int(t // window_ns)


def request_stage_windows(telemetry_dir, window_s):
    """Per (window, instance) request-stage aggregates (observable).

    Decode-side rows are attributed to the request's end time, prefill-side
    rows to the prefill completion time. The two are merged by the caller,
    so a prefill window carries prefill service time and a decode window
    carries TTFT/TPOT/ITL/queue.
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
    defaults = {
        "req_count": 0.0, "req_ttft_mean": -1.0, "req_tpot_mean": -1.0,
        "req_itl_p99_mean": -1.0, "req_queue_mean": -1.0,
        "req_prefill_svc_mean": -1.0,
    }
    window.update(defaults)
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
    from the request telemetry (observable). ``role`` is a legacy fallback
    for single-role unit tests and is only used when inference yields
    nothing.

    Returns a list of dicts, one per (window_end_s, instance_id), with
    observable-only statistics. Windows with no batches are skipped.
    """
    path = os.path.join(telemetry_dir, "batch_telemetry.csv")
    if not os.path.exists(path):
        return []
    rows = read_csv_rows(path)
    if roles is None:
        roles = infer_instance_roles(telemetry_dir)
    window_ns = int(window_s * 1e9)
    buckets = {}
    for row, multiplicity in _dedupe_batches(rows):
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
            "dur_p99": float(np.percentile(durations, 99)) if len(durations) > 1 else float(durations[0]),
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
