"""Operational regime of a completed run, from observable telemetry only.

The reviewer's point (Q3 / P1 item 9) is that a detection number is
meaningless without the regime it was measured in: a 1.1x fault is masked
by queueing at 99% utilisation and diluted by idle time at 19%. This
module turns a finished run's telemetry into the four numbers that define
its regime, and `generate.py` writes them into every manifest entry.

Utilisation is busy time over the run span, computed from DEDUPLICATED
batch durations (one per (instance, batch)), so a TP=2 instance is not
counted twice. It is a per-instance lower bound on utilisation: batches
that overlap are summed, then capped at 1.0.
"""

import os

import numpy as np

from experiments.fault_detection.ml.features import (
    _as_float,
    _dedupe_batches,
    infer_instance_roles,
    read_csv_rows,
)


def _p50(values):
    values = [v for v in values if v >= 0]
    return float(np.median(values)) if values else -1.0


def regime_from_telemetry(telemetry_dir):
    """Return the regime dict for one run directory (never raises)."""
    empty = {"lambda_sps": -1.0, "u_prefill": -1.0, "u_decode": -1.0,
             "tpot_p50_ms": -1.0, "span_s": -1.0, "n_requests": 0}
    batch_path = os.path.join(telemetry_dir, "batch_telemetry.csv")
    req_path = os.path.join(telemetry_dir, "request_telemetry.csv")
    if not os.path.exists(batch_path):
        return empty
    rows = read_csv_rows(batch_path)
    batches = [r for r, _m in _dedupe_batches(rows)]
    if not batches:
        return empty
    roles = infer_instance_roles(telemetry_dir)
    start = min(_as_float(r, "batch_time_ns") for r in batches)
    finish = max(_as_float(r, "finish_ns") for r in batches)
    span = (finish - start) / 1e9
    if span <= 0:
        return empty
    busy = {}
    for r in batches:
        inst = int(_as_float(r, "instance_id"))
        busy[inst] = busy.get(inst, 0.0) + _as_float(r, "duration_ns") / 1e9
    by_role = {}
    for inst, seconds in busy.items():
        role = roles.get(inst, "unknown")
        by_role.setdefault(role, []).append(min(1.0, seconds / span))
    n_requests = 0
    tpot = -1.0
    if os.path.exists(req_path):
        reqs = read_csv_rows(req_path)
        n_requests = len(reqs)
        tpot = _p50([_as_float(r, "tpot", -1) for r in reqs]) / 1e6
    lam = n_requests / span if n_requests else -1.0
    return {
        "lambda_sps": lam,
        "u_prefill": float(max(by_role.get("prefill", [0.0]))),
        "u_decode": float(max(by_role.get("decode", by_role.get("colocated", [0.0])))),
        "tpot_p50_ms": tpot,
        "span_s": span,
        "n_requests": n_requests,
    }
