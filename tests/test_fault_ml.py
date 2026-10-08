"""Tests for the fault-detection ML scaffolding. No binary needed."""

import inspect
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from experiments.fault_detection.manifest import MANIFEST_ONLY_COLUMNS
from experiments.fault_detection.ml import features


def _row(**kw):
    base = {"duration_ns": "1000", "total_len": "64", "num_prefill": "1",
            "n_decode": "4", "kv_len": "64", "pd_kv_send_tokens": "0",
            "com_cycle": "10", "n_waiting": "0", "n_running": "1",
            "batch_time_ns": "1000000000", "instance_id": "0"}
    base.update(kw)
    return base


def test_no_manifest_column_reachable_from_feature_builder():
    import re
    source = inspect.getsource(features)
    # Blank out the FORBIDDEN_COLUMNS ban declaration itself; nothing
    # else in the module may mention a ground-truth column name.
    stripped = re.sub(r"FORBIDDEN_COLUMNS = frozenset\(\{.*?\}\)", "",
                      source, flags=re.S)
    stripped = stripped.replace("FORBIDDEN_COLUMNS", "")
    leaked = [c for c in MANIFEST_ONLY_COLUMNS if c in stripped]
    assert leaked == [], "feature builder references ground truth: %s" % leaked
    assert features.FORBIDDEN_COLUMNS >= MANIFEST_ONLY_COLUMNS


def test_residual_model_fits_healthy_only(tmp_path):
    import csv
    path = tmp_path / "batch_telemetry.csv"
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(_row().keys()))
        writer.writeheader()
        for i in range(10):
            writer.writerow(_row(duration_ns=str(1000 + i * 10),
                                 total_len=str(64 + i),
                                 batch_time_ns=str((i + 1) * 1_000_000_000)))
    windows = features.build_windows(str(tmp_path), window_s=20.0, role="decode")
    assert len(windows) == 1
    model = features.HealthyResidualModel()
    assert model.fit([_row(duration_ns=str(1000 + i * 10), total_len=str(64 + i))
                      for i in range(10)]) is True
    resid = model.residual({**windows[0], "dur_residual": 0.0,
                            "com_residual": 0.0})
    assert abs(resid) < 500.0


def test_observable_guard_rejects_leaked_columns(tmp_path):
    import csv
    from experiments.fault_detection.manifest import check_observable_clean
    path = tmp_path / "batch_telemetry.csv"
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["run_id", "fault_present", "dur_mean"])
        writer.writerow(["r1", "True", "1.0"])
    import pytest
    with pytest.raises(ValueError):
        check_observable_clean(str(tmp_path))
