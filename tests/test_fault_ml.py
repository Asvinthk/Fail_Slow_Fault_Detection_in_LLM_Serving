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
            "batch_time_ns": "1000000000", "instance_id": "0",
            "batch_id": "1", "sys": "0"}
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
                                 batch_id=str(i + 1),
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


# ---------------------------------------------------------------------------
# Review fixes: phantom rows (F7), per-NPU dedupe (F8), per-instance roles
# (F2), leave-one-out peers (N3), value-level leak scan (N6).
# ---------------------------------------------------------------------------

BATCH_HEADER = ("run_id,instance_id,batch_id,sys,batch_time_ns,finish_ns,"
                "duration_ns,num_prefill,num_decode,total_len,kv_len,"
                "pd_kv_send_tokens,n_waiting,n_running,com_cycle\n")


def _write_batch(tmp_path, lines, requests=None):
    with open(tmp_path / "batch_telemetry.csv", "w") as f:
        f.write(BATCH_HEADER)
        f.writelines(lines)
    if requests is not None:
        with open(tmp_path / "request_telemetry.csv", "w") as f:
            f.write("request_id,prefill_instance_id,decode_instance_id,"
                    "input_toks,output_toks,generated_toks,arrival_ns,"
                    "first_scheduled_ns,prefill_done_ns,decode_admit_ns,"
                    "first_decode_done_ns,ttft_ns,latency_ns,queuing_delay_ns,"
                    "tpot,itl_count,itl_mean_ns,itl_p50_ns,itl_p99_ns,itl_max_ns\n")
            f.writelines(requests)
    return str(tmp_path)


def _line(inst, batch_id, dur_ns, sys_id=None, finish_ns=1_500_000_000,
          total_len=64):
    return "run_x,%d,%d,%d,%d,%d,%d,1,0,%d,64,0,0,1,0\n" % (
        inst, batch_id, inst if sys_id is None else sys_id,
        finish_ns - dur_ns, finish_ns, dur_ns, total_len)


def test_phantom_startup_rows_are_dropped(tmp_path):
    # batch_id 0 is the start-up report every NPU emits before any batch.
    lines = [_line(0, 0, 0, total_len=0), _line(0, 1, 1000)]
    d = _write_batch(tmp_path, lines)
    windows = features.build_windows(d, window_s=2.0, roles={0: "prefill"})
    assert len(windows) == 1
    assert windows[0]["count"] == 1
    assert windows[0]["dur_mean"] == 1000.0


def test_rows_are_deduped_per_instance_batch(tmp_path):
    # One batch reported by two NPUs of the same instance (TP=2 decode, or
    # a PD prefill sender + shadow receiver) must count as ONE batch.
    lines = [_line(0, 1, 1000, sys_id=0), _line(0, 1, 1000, sys_id=4)]
    d = _write_batch(tmp_path, lines)
    windows = features.build_windows(d, window_s=2.0, roles={0: "decode"})
    assert windows[0]["count"] == 1
    assert windows[0]["throughput_per_s"] == 0.5  # batches/s, not rows/s
    assert windows[0]["npus_reporting_mean"] == 2.0


def test_roles_are_resolved_per_instance(tmp_path):
    requests = [
        "0,0,1,10,20,10,0,0,0,0,0,1000,2000,0,1,0,1,1,1,1\n",
        "1,0,1,10,20,10,0,0,0,0,0,1000,2000,0,1,0,1,1,1,1\n",
    ]
    lines = [_line(0, 1, 1000), _line(1, 1, 1000)]
    d = _write_batch(tmp_path, lines, requests=requests)
    windows = features.build_windows(d, window_s=2.0)  # no role argument
    roles = {w["instance_id"]: w["role"] for w in windows}
    assert roles == {0: "prefill", 1: "decode"}


def test_peer_ratio_is_leave_one_out(tmp_path):
    # Three decode instances; the third is 2x slower. A mean-including-self
    # reference would give 20/13.3 = 1.5; the LOO median gives 2.0.
    lines = [_line(0, 1, 10_000_000), _line(1, 1, 10_000_000),
             _line(2, 1, 20_000_000)]
    d = _write_batch(tmp_path, lines)
    roles = {0: "decode", 1: "decode", 2: "decode"}
    windows = features.build_windows(d, window_s=2.0, roles=roles)
    features.add_peer_and_residual_features([windows], features.HealthyResidualModel())
    slow = [w for w in windows if w["instance_id"] == 2][0]
    assert slow["has_peer"] == 1.0
    assert abs(slow["dur_peer_ratio"] - 2.0) < 1e-9


def test_lone_instance_has_no_peer(tmp_path):
    # 1P1D: one decode instance, so peer features are undefined and must be
    # flagged, not silently reported as a meaningful 1.0 (review N3).
    d = _write_batch(tmp_path, [_line(0, 1, 1000)])
    windows = features.build_windows(d, window_s=2.0, roles={0: "decode"})
    features.add_peer_and_residual_features([windows], features.HealthyResidualModel())
    assert windows[0]["has_peer"] == 0.0
    assert windows[0]["dur_peer_ratio"] == 1.0


def test_observable_guard_rejects_leaked_values(tmp_path):
    import pytest
    from experiments.fault_detection.manifest import check_observable_clean
    _write_batch(tmp_path, [_line(0, 1, 1000)])
    with open(tmp_path / "batch_telemetry.csv") as f:
        text = f.read()
    with open(tmp_path / "batch_telemetry.csv", "w") as f:
        f.write(text.replace("run_x", "pair42_faulty"))
    with pytest.raises(ValueError):
        check_observable_clean(str(tmp_path))


def test_window_dump_columns_are_not_ground_truth():
    # The per-window dump may carry the label (it is the evaluation dump),
    # but the FEATURE vector must not.
    assert not (set(features.FEATURE_COLUMNS) & MANIFEST_ONLY_COLUMNS)
    assert "run_id" not in features.FEATURE_COLUMNS
