"""Per-window, INSTANCE-level ground truth derived from the audit.

Evaluation-only. This module is the one place allowed to read
injection_audit.jsonl and the manifest; `ml/features.py` must never
import it (enforced by test_fault_ml.test_no_manifest_column_reachable).

Why this exists (review item F3 + N1 + N7 + N8):

  * the old label was derived from the fault SCHEDULE and a single
    run-level `successfully_applied` flag, so every window inside the
    interval was "anomalous" -- including the off-duty half of an
    intermittent fault and including instances the fault never touched;
  * windows were bucketed by schedule time but labelled by window end,
    so a batch scheduled before onset and finishing inside the window
    was mislabelled (N1). Both sides now use finish_ns;
  * there was no notion of a downstream victim or of post-fault
    recovery, so both were silently counted as false positives (N7/N8);
  * the label carried no instance, so M3 (localisation) could not be
    scored at all. Labels are now `decode:1`, `kv-transfer-network:0`,
    `worker-unavailable:2`, ...

Label kinds and how the evaluation must treat them:

  root_cause   -- the audit shows the mechanism applied to THIS instance
                  in THIS window (share of its batches >= applied_share).
                  Counts as a positive.
  neutral      -- the fault was active on this instance but every audit
                  record says applied=false (mechanism neutral for the
                  workload, e.g. a KV-transfer fault on a window that
                  shipped no KV). Reported separately, excluded from the
                  recall denominator (review Q10).
  downstream   -- some other instance was affected in this window and
                  this one was not. Excluded from both TP and FP counts,
                  because a slow decode legitimately backs up prefill
                  (N8). Counted and reported on its own.
  recovering   -- inside [fault_end, fault_end + recovery_s) with no
                  fault activity: the queue backlog outlives the fault.
                  Excluded from the FPR denominator (N7).
  unaffected   -- everything else. The only FPR denominator.
"""

import os

from experiments.fault_detection.manifest import (
    expected_label_for_fault,
    load_audit,
)
from experiments.fault_detection.ml.features import (
    _as_float,
    _dedupe_batches,
    _window_index,
    read_csv_rows,
)

DEFAULT_APPLIED_SHARE = 0.5
DEFAULT_RECOVERY_S = 2.0

KIND_ROOT = "root_cause"
KIND_NEUTRAL = "neutral"
KIND_DOWNSTREAM = "downstream"
KIND_RECOVERING = "recovering"
KIND_UNAFFECTED = "unaffected"

# batch_telemetry.csv's `batch_id` column is the ASTRA-Sim REPORT id, which
# is the scheduler's batch id + 1 (`add_done` subtracts 1 before looking the
# batch up in batch_meta). injection_audit.jsonl records the scheduler's
# batch id. Joining the two without this offset silently shifts every label
# by one batch, so the offset is named and asserted in the tests.
REPORT_ID_OFFSET = 1

# Kinds that enter the binary detection metric.
POSITIVE_KINDS = (KIND_ROOT,)
NEGATIVE_KINDS = (KIND_UNAFFECTED,)
# Kinds excluded from both sides of the metric (reported separately).
EXCLUDED_KINDS = (KIND_DOWNSTREAM, KIND_RECOVERING, KIND_NEUTRAL)


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

    `entry` is the manifest record for the run (fault specs, schedule).
    A healthy run yields only `unaffected` labels.
    """
    window_ns = int(window_s * 1e9)
    labels = {}

    # Batch inventory per (window, instance), from observable telemetry.
    path = os.path.join(telemetry_dir, "batch_telemetry.csv")
    rows = read_csv_rows(path) if os.path.exists(path) else []
    batch_window = {}
    counts = {}
    for row, _multiplicity in _dedupe_batches(rows):
        inst = int(_as_float(row, "instance_id"))
        widx = _window_index(row, window_ns)
        bid = int(_as_float(row, "batch_id"))
        batch_window[(inst, bid)] = widx
        counts[(widx, inst)] = counts.get((widx, inst), 0) + 1

    # Audit -> per-(window, instance) applied / neutral batch counts.
    applied = {}
    neutral = {}
    event_keyed = set()
    for record in load_audit(telemetry_dir):
        inst = record.get("instance")
        bid = record.get("batch_id")
        if inst is None:
            continue
        if bid is None:
            # Event-keyed records: a kv_handoff_delay hold, or a
            # worker_stall window summary. They carry no batch, so they
            # are attributed to the window containing the event time.
            # The hold is a transfer-link event owned by the prefill
            # instance, so the label becomes kv-transfer-network:<prefill>.
            if record.get("applied") and record.get("time_ns") is not None:
                widx = int(float(record["time_ns"]) // window_ns)
                applied[(widx, int(inst))] = applied.get((widx, int(inst)), 0) + 1
                event_keyed.add((widx, int(inst)))
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
            labels[_window_key(window_end_s, inst)] = {
                "kind": KIND_ROOT,
                "component": component or "unknown",
                "instance_id": inst,
                "label": "%s:%d" % (component or "unknown", inst),
                "applied_share": share,
                "partial": share < applied_share,
                "batches": batches,
                "window_end_s": window_end_s,
            }
        elif z:
            labels[_window_key(window_end_s, inst)] = {
                "kind": KIND_NEUTRAL,
                "component": component or "unknown",
                "instance_id": inst,
                "label": "%s:%d" % (component or "unknown", inst),
                "applied_share": 0.0,
                "partial": False,
                "batches": batches,
                "window_end_s": window_end_s,
            }
        elif widx in active_windows:
            labels[_window_key(window_end_s, inst)] = {
                "kind": KIND_DOWNSTREAM,
                "component": "downstream",
                "instance_id": inst,
                "label": "downstream:%d" % inst,
                "applied_share": 0.0,
                "partial": False,
                "batches": batches,
                "window_end_s": window_end_s,
            }
        elif widx in recovery_windows:
            labels[_window_key(window_end_s, inst)] = {
                "kind": KIND_RECOVERING,
                "component": "recovering",
                "instance_id": inst,
                "label": "recovering:%d" % inst,
                "applied_share": 0.0,
                "partial": False,
                "batches": batches,
                "window_end_s": window_end_s,
            }
        else:
            labels[_window_key(window_end_s, inst)] = {
                "kind": KIND_UNAFFECTED,
                "component": "healthy",
                "instance_id": inst,
                "label": "unaffected",
                "applied_share": 0.0,
                "partial": False,
                "batches": batches,
                "window_end_s": window_end_s,
            }
    return labels


def attach_labels(windows, labels):
    """Attach each window's label dict; windows without a label are dropped.

    A window with no audit-derived label cannot be scored, so it is
    reported as `unlabelled` rather than silently treated as healthy.
    """
    out = []
    for window in windows:
        key = _window_key(window["window_end_s"], window["instance_id"])
        lab = labels.get(key)
        if lab is None:
            lab = {
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
