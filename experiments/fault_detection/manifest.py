"""Manifest loader and ground-truth guards for fault-detection experiments.

manifest.jsonl has one line per run: experiment_id, run_id, pair_id,
workload_id + parameters, cluster_config_id, seed, git commit,
fault_present, per-fault detail (start/end, target role + instance_id,
mode, injection_layer, severity/parameters, failure_class,
successfully_applied), expected component-level label, run status
(ok/failed + reason), and Total clocks.

Ground truth lives ONLY in injection_audit.jsonl + the manifest and
must never leak into observable telemetry. check_observable_clean()
fails a run directory whose telemetry headers contain manifest-only
columns. check_audit_consistency() verifies each faulty run's audit
lies within its manifest interval and target, and healthy runs have
empty audits.
"""

import csv
import json
import os

# Columns that must never appear in observable telemetry files.
MANIFEST_ONLY_COLUMNS = frozenset({
    "fault_present", "fault_id", "failure_class", "successfully_applied",
    "pair_id", "expected_label", "workload_id", "experiment_id",
})

MODE_TO_FAILURE_CLASS = {
    "compute_multiplicative": "fail_slow",
    "compute_additive": "fail_slow",
    "kv_transfer_bandwidth": "fail_slow",
    "kv_handoff_delay": "fail_slow",
    "network_degradation": "fail_slow",
    "worker_stall": "fail_stop_stall",
}

MODE_TO_COMPONENT = {
    "compute_multiplicative": None,  # resolved by target role below
    "compute_additive": None,
    "kv_transfer_bandwidth": "kv-transfer-network",
    "kv_handoff_delay": "kv-transfer-network",
    "network_degradation": "kv-transfer-network",
    "worker_stall": "worker-unavailable",
}


def expected_label_for_fault(fault):
    component = MODE_TO_COMPONENT.get(fault.get("mode"))
    if component is None:
        role = fault.get("target", {}).get("role")
        if role == "prefill":
            component = "prefill"
        elif role == "decode":
            component = "decode"
        elif role in ("transfer", "network"):
            component = "kv-transfer-network"
        else:
            component = "unknown"
    if fault.get("mode") == "worker_stall" and fault.get("target", {}).get("role") in ("prefill", "decode"):
        # Stall localizes to the stalled role's pool; keep the
        # worker-unavailable class so it is not confused with fail-slow.
        component = "worker-unavailable"
    return component


def load_manifest(path):
    entries = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def check_observable_clean(telemetry_dir):
    """Raise ValueError if an observable file has manifest-only columns."""
    for name in ("batch_telemetry.csv", "request_telemetry.csv"):
        path = os.path.join(telemetry_dir, name)
        if not os.path.exists(path):
            continue
        with open(path) as f:
            header = next(csv.reader(f), [])
        leaked = [c for c in header if c in MANIFEST_ONLY_COLUMNS]
        if leaked:
            raise ValueError("observable file %s leaks ground truth columns: %s"
                             % (path, leaked))


def audit_within_interval(audit_records, fault):
    """True if at least one audit record for fault_id is inside [start, end)."""
    schedule = fault.get("schedule", {})
    start_ns = schedule.get("start_s", 0) * 1e9
    end_ns = schedule.get("end_s", float("inf")) * 1e9
    for record in audit_records:
        if record.get("fault_id") != fault.get("fault_id"):
            continue
        time_ns = record.get("time_ns", 0)
        if start_ns <= time_ns < end_ns:
            return True, ""
    return False, "no audit record for %s inside [%s, %s)s" % (
        fault.get("fault_id"), schedule.get("start_s"), schedule.get("end_s"))


def check_audit_consistency(entry, audit_records):
    """Verify audit/manifest agreement for one manifest entry.

    Returns (ok, reason). Faulty runs need >=1 in-interval audit record
    per fault; healthy runs need empty audits.
    """
    faults = entry.get("faults", [])
    if not entry.get("fault_present"):
        if audit_records:
            return False, "healthy run has %d audit records" % len(audit_records)
        return True, ""
    for fault in faults:
        ok, reason = audit_within_interval(audit_records, fault)
        if not ok:
            return False, reason
    return True, ""


def load_audit(telemetry_dir):
    path = os.path.join(telemetry_dir, "injection_audit.jsonl")
    if not os.path.exists(path):
        return []
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records
