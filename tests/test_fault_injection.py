"""Unit tests for the fail-slow prototype. No ASTRA-Sim binary needed.

Run: python -m pytest tests/test_fault_injection.py -v
Tests that need the binary are skipped with a clear marker elsewhere.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from serving.core.fault_injection import (
    severity_weight,
    validate_fault_config,
    fault_matches,
    is_stalled,
    apply_to_trace_rows,
)


def make_row(layer="attention_0", comp=1000, comm=0):
    return [layer, str(comp), "LOCAL", "10", "LOCAL", "10",
            "LOCAL", "10", "NONE", str(comm), "NONE"]


def test_step_weight():
    schedule = {"start_s": 1.0, "end_s": 2.0, "profile": "step"}
    assert severity_weight(schedule, 0.5) == 0.0
    assert severity_weight(schedule, 1.5) == 1.0
    assert severity_weight(schedule, 2.5) == 0.0


def test_window_before_and_after_unaffected():
    schedule = {"start_s": 1.0, "end_s": 2.0, "profile": "step"}
    faults = [{"fault_id": "f1", "target": {"role": "decode", "instance_id": 1},
               "injection_layer": "mechanism", "mode": "compute_multiplicative",
               "factor": 2.0, "schedule": schedule}]
    rows_before = [make_row(comp=1000)]
    audits = apply_to_trace_rows(faults, rows_before, "decode", 1, 0.5e9)
    assert audits == []
    assert rows_before[0][1] == "1000"


def test_multiplicative_scales_target_only():
    schedule = {"start_s": 0.0, "end_s": 10.0, "profile": "step"}
    faults = [{"fault_id": "f1", "target": {"role": "decode", "instance_id": 1},
               "injection_layer": "mechanism", "mode": "compute_multiplicative",
               "factor": 2.0, "schedule": schedule}]
    rows = [make_row(comp=1000)]
    audits = apply_to_trace_rows(faults, rows, "decode", 1, 5.0)
    assert rows[0][1] == "2000"
    assert audits[0]["applied"] is True
    other = [make_row(comp=1000)]
    audits_other = apply_to_trace_rows(faults, other, "decode", 0, 5.0)
    assert audits_other == []
    assert other[0][1] == "1000"


def test_tuple_rows_are_copied_before_scaling():
    # generate_trace builds rows as tuples; scaling must not crash.
    schedule = {"start_s": 0.0, "end_s": 10.0, "profile": "step"}
    faults = [{"fault_id": "f1", "target": {"role": "decode", "instance_id": 1},
               "injection_layer": "mechanism", "mode": "compute_multiplicative",
               "factor": 2.0, "schedule": schedule}]
    rows = [tuple(make_row(comp=1000))]
    audits = apply_to_trace_rows(faults, rows, "decode", 1, 5.0)
    assert rows[0][1] == "2000"
    assert audits[0]["applied"] is True


def test_additive_delta():
    schedule = {"start_s": 0.0, "end_s": 10.0, "profile": "step"}
    faults = [{"fault_id": "f1", "target": {"role": "prefill", "instance_id": 0},
               "injection_layer": "mechanism", "mode": "compute_additive",
               "delta_ns": 500, "schedule": schedule}]
    rows = [make_row(comp=1000)]
    apply_to_trace_rows(faults, rows, "prefill", 0, 5.0)
    assert rows[0][1] == "1500"


def test_kv_bandwidth_inflates_comm_size():
    schedule = {"start_s": 0.0, "end_s": 10.0, "profile": "step"}
    faults = [{"fault_id": "f1", "target": {"role": "transfer", "instance_id": 0},
               "injection_layer": "mechanism", "mode": "kv_transfer_bandwidth",
               "bandwidth_fraction": 0.5, "schedule": schedule}]
    rows = [make_row(layer="qkv_proj_0", comp=1000, comm=800)]
    apply_to_trace_rows(faults, rows, "prefill", 0, 5.0)
    assert rows[0][9] == "1600"


def test_intermittent_and_ramp():
    inter = {"start_s": 0.0, "end_s": 10.0, "profile": "intermittent",
             "period_s": 2.0, "duty": 0.5}
    assert severity_weight(inter, 0.5) == 1.0
    assert severity_weight(inter, 1.5) == 0.0
    ramp = {"start_s": 0.0, "end_s": 4.0, "profile": "ramp_up", "ramp_s": 4.0}
    assert severity_weight(ramp, 2.0) == pytest.approx(0.5)


def test_stall_and_validation():
    schedule = {"start_s": 0.0, "end_s": 10.0, "profile": "step"}
    faults = [{"fault_id": "s1", "target": {"role": "decode", "instance_id": 0},
               "injection_layer": "mechanism", "mode": "worker_stall",
               "schedule": schedule}]
    assert is_stalled(faults, "decode", 0, 5.0) is True
    assert is_stalled(faults, "decode", 1, 5.0) is False
    with pytest.raises(ValueError):
        validate_fault_config({"faults": [{"fault_id": "x", "target": {"role": "decode", "instance_id": 0},
                                            "injection_layer": "mechanism", "mode": "nope",
                                            "schedule": schedule}]})
    with pytest.raises(ValueError):
        validate_fault_config({"faults": [], "bogus_key": 1})
