"""Tests for audit-derived, per-window, per-instance labels (review F3/N1/N8).

No simulator binary needed: the telemetry and audit files are synthetic,
but they use the real column names and the real report-id offset.
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from experiments.fault_detection.labels import (  # noqa: E402
    KIND_DOWNSTREAM,
    KIND_NEUTRAL,
    KIND_RECOVERING,
    KIND_ROOT,
    KIND_UNAFFECTED,
    REPORT_ID_OFFSET,
    label_windows,
)
from experiments.fault_detection.ml.features import build_windows  # noqa: E402

HEADER = ("run_id,instance_id,batch_id,sys,batch_time_ns,finish_ns,duration_ns,"
          "num_prefill,num_decode,total_len,kv_len,pd_kv_send_tokens,"
          "n_waiting,n_running,com_cycle\n")


def _batch(inst, report_id, finish_s, dur_ns=10_000_000, sys_id=None):
    return "%s,%d,%d,%d,%d,%d,%d,1,0,64,64,0,0,1,0\n" % (
        "run_x", inst, report_id, inst if sys_id is None else sys_id,
        int((finish_s - 0.1) * 1e9), int(finish_s * 1e9), dur_ns)


def _write(tmp_path, rows, audit=None, requests=None):
    with open(tmp_path / "batch_telemetry.csv", "w") as f:
        f.write(HEADER)
        f.writelines(rows)
    if audit is not None:
        with open(tmp_path / "injection_audit.jsonl", "w") as f:
            for record in audit:
                f.write(json.dumps(record) + "\n")
    if requests is not None:
        with open(tmp_path / "request_telemetry.csv", "w") as f:
            f.write("request_id,prefill_instance_id,decode_instance_id,"
                    "input_toks,output_toks,generated_toks,arrival_ns,"
                    "first_scheduled_ns,prefill_done_ns,decode_admit_ns,"
                    "first_decode_done_ns,ttft_ns,latency_ns,queuing_delay_ns,"
                    "tpot,itl_count,itl_mean_ns,itl_p50_ns,itl_p99_ns,itl_max_ns\n")
            f.writelines(requests)
    return str(tmp_path)


ENTRY = {
    "fault_present": True,
    "successfully_applied": True,
    "faults": [{
        "fault_id": "f0",
        "mode": "compute_multiplicative",
        "target": {"role": "decode", "instance_id": 1},
        "schedule": {"start_s": 0.0, "end_s": 2.0},
    }],
}


def test_label_depends_on_instance_and_uses_audit(tmp_path):
    # Two instances, four batches. Only decode instance 1 is affected, and
    # only for the batch whose report id is audit batch_id + 1.
    rows = [_batch(0, 1, 1.0), _batch(0, 2, 1.5),
            _batch(1, 1, 1.2), _batch(1, 2, 1.8)]
    audit = [{"fault_id": "f0", "instance": 1, "batch_id": 1, "applied": True,
              "time_ns": 1_000_000_000}]
    d = _write(tmp_path, rows, audit)
    labels = label_windows(d, ENTRY, window_s=2.0)

    assert REPORT_ID_OFFSET == 1
    decode = labels[(2.0, 1)]
    prefill = labels[(2.0, 0)]
    assert decode["kind"] == KIND_ROOT
    assert decode["label"] == "decode:1"
    assert decode["applied_share"] == 0.5  # 1 of 2 decode batches
    # The prefill instance is not affected by a decode fault: it is a
    # downstream victim, never a positive and never a false-positive
    # candidate.
    assert prefill["kind"] == KIND_DOWNSTREAM


def test_windows_after_fault_are_recovering_not_healthy(tmp_path):
    rows = [_batch(1, 1, 1.0), _batch(1, 2, 3.0), _batch(1, 3, 5.0),
            _batch(1, 4, 7.0)]
    audit = [{"fault_id": "f0", "instance": 1, "batch_id": 0, "applied": True,
              "time_ns": 1_000_000_000}]
    d = _write(tmp_path, rows, audit)
    labels = label_windows(d, ENTRY, window_s=2.0, recovery_s=2.0)
    assert labels[(2.0, 1)]["kind"] == KIND_ROOT        # fault active
    assert labels[(4.0, 1)]["kind"] == KIND_RECOVERING  # inside recovery
    assert labels[(6.0, 1)]["kind"] == KIND_RECOVERING  # still inside
    assert labels[(8.0, 1)]["kind"] == KIND_UNAFFECTED  # after recovery


def test_neutral_mechanism_is_its_own_kind(tmp_path):
    rows = [_batch(1, 1, 1.0)]
    audit = [{"fault_id": "f0", "instance": 1, "batch_id": 0, "applied": False,
              "time_ns": 1_000_000_000}]
    d = _write(tmp_path, rows, audit)
    labels = label_windows(d, ENTRY, window_s=2.0)
    assert labels[(2.0, 1)]["kind"] == KIND_NEUTRAL


def test_event_keyed_handoff_labels_the_transfer_link(tmp_path):
    """kv_handoff_delay writes hold records with no batch id.

    They are attributed by event time to the prefill instance that owns
    the link, so a handoff fault is still a root cause (and the decode
    instance that waited is a downstream victim).
    """
    rows = [_batch(0, 1, 1.0), _batch(1, 1, 1.2)]
    audit = [{"type": "handoff_hold", "fault_id": "handoff",
              "mode": "kv_handoff_delay", "instance": 0,
              "time_ns": 1_000_000_000, "applied": True}]
    entry = {"fault_present": True, "faults": [{
        "fault_id": "handoff", "mode": "kv_handoff_delay",
        "target": {"role": "transfer", "instance_id": 0},
        "schedule": {"start_s": 1.0, "end_s": 2.0}}]}
    d = _write(tmp_path, rows, audit)
    labels = label_windows(d, entry, window_s=2.0)
    assert labels[(2.0, 0)]["kind"] == KIND_ROOT
    assert labels[(2.0, 0)]["label"] == "kv-transfer-network:0"
    assert labels[(2.0, 1)]["kind"] == KIND_DOWNSTREAM


def test_healthy_run_is_all_unaffected(tmp_path):
    rows = [_batch(0, 1, 1.0), _batch(1, 1, 1.5)]
    d = _write(tmp_path, rows, audit=[])
    labels = label_windows(d, {"fault_present": False, "faults": []}, window_s=2.0)
    assert {lab["kind"] for lab in labels.values()} == {KIND_UNAFFECTED}


def test_labels_attach_to_matching_windows(tmp_path):
    rows = [_batch(0, 1, 1.0), _batch(1, 1, 1.2)]
    requests = ["0,0,1,10,20,10,0,0,0,0,0,1000,2000,0,1,0,1,1,1,1\n"]
    d = _write(tmp_path, rows, audit=[], requests=requests)
    windows = build_windows(d, window_s=2.0)
    from experiments.fault_detection.labels import attach_labels
    labelled = attach_labels(windows, label_windows(d, {"faults": []}, window_s=2.0))
    assert len(labelled) == len(windows)
    assert all(w["label_kind"] == KIND_UNAFFECTED for w in labelled)
    # Roles came from the request telemetry, not from a single argument.
    roles = {w["instance_id"]: w["role"] for w in labelled}
    assert roles == {0: "prefill", 1: "decode"}
