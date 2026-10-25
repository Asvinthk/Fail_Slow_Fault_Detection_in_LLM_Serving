"""Host-only tests for experiments/fault_detection/evaluate.py.

Merged from test_fault_ml.py and test_labels.py. The synthetic telemetry
uses the real column names and the real report-id offset, so the join
semantics under test are the ones the simulator produces.
"""

import inspect
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from experiments.fault_detection import evaluate  # noqa: E402
from experiments.fault_detection.evaluate import (  # noqa: E402
    KIND_DOWNSTREAM,
    KIND_NEUTRAL,
    KIND_RECOVERING,
    KIND_ROOT,
    KIND_UNAFFECTED,
    MANIFEST_ONLY_COLUMNS,
    REPORT_ID_OFFSET,
    attach_labels,
    build_windows,
    label_windows,
)


def _row(**kw):
    base = {"duration_ns": "1000", "total_len": "64", "num_prefill": "1",
            "n_decode": "4", "kv_len": "64", "pd_kv_send_tokens": "0",
            "com_cycle": "10", "n_waiting": "0", "n_running": "1",
            "batch_time_ns": "1000000000", "instance_id": "0",
            "batch_id": "1", "sys": "0"}
    base.update(kw)
    return base


# ---------------------------------------------------------------------------
# observable-only contract
# ---------------------------------------------------------------------------
def test_no_ground_truth_reachable_from_feature_builder():
    """The feature path must not read a manifest/audit column name.

    The scan is scoped to the feature-building functions rather than the
    whole module: evaluate.py also contains the labelling code, which
    legitimately reads `fault_id` from the audit. Scoping is what makes the
    assertion meaningful instead of vacuous.
    """
    funcs = [evaluate.build_windows, evaluate.dedupe_batches,
             evaluate._is_phantom, evaluate._window_index,
             evaluate.infer_instance_roles, evaluate.request_stage_windows,
             evaluate._merge_request_stage, evaluate.HealthyResidualModel,
             evaluate.add_peer_and_residual_features,
             evaluate.window_to_vector]
    source = "".join(inspect.getsource(f) for f in funcs)
    leaked = [c for c in MANIFEST_ONLY_COLUMNS if c in source]
    assert leaked == [], "feature builder references ground truth: %s" % leaked
    assert evaluate.FORBIDDEN_COLUMNS >= MANIFEST_ONLY_COLUMNS


def test_feature_vector_carries_no_label_or_run_identity():
    assert not (set(evaluate.FEATURE_COLUMNS) & MANIFEST_ONLY_COLUMNS)
    assert "run_id" not in evaluate.FEATURE_COLUMNS
    assert "label" not in evaluate.FEATURE_COLUMNS


# ---------------------------------------------------------------------------
# residual model and window aggregation
# ---------------------------------------------------------------------------
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
    windows = build_windows(str(tmp_path), window_s=20.0, role="decode")
    assert len(windows) == 1
    model = evaluate.HealthyResidualModel()
    assert model.fit([_row(duration_ns=str(1000 + i * 10), total_len=str(64 + i))
                      for i in range(10)]) is True
    resid = model.residual({**windows[0], "dur_residual": 0.0,
                            "com_residual": 0.0})
    assert abs(resid) < 500.0


# ---------------------------------------------------------------------------
# telemetry hygiene: phantom rows (F7), per-NPU dedupe (F8)
# ---------------------------------------------------------------------------
BATCH_HEADER = ("run_id,instance_id,batch_id,sys,batch_time_ns,finish_ns,"
                "duration_ns,num_prefill,num_decode,total_len,kv_len,"
                "pd_kv_send_tokens,n_waiting,n_running,com_cycle\n")

REQUEST_HEADER = ("request_id,prefill_instance_id,decode_instance_id,"
                  "input_toks,output_toks,generated_toks,arrival_ns,"
                  "first_scheduled_ns,prefill_done_ns,decode_admit_ns,"
                  "first_decode_done_ns,ttft_ns,latency_ns,queuing_delay_ns,"
                  "tpot,itl_count,itl_mean_ns,itl_p50_ns,itl_p99_ns,itl_max_ns\n")


def _write_batch(tmp_path, lines, requests=None, audit=None):
    with open(tmp_path / "batch_telemetry.csv", "w") as f:
        f.write(BATCH_HEADER)
        f.writelines(lines)
    if requests is not None:
        with open(tmp_path / "request_telemetry.csv", "w") as f:
            f.write(REQUEST_HEADER)
            f.writelines(requests)
    if audit is not None:
        with open(tmp_path / "injection_audit.jsonl", "w") as f:
            for record in audit:
                f.write(json.dumps(record) + "\n")
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
    windows = build_windows(d, window_s=2.0, roles={0: "prefill"})
    assert len(windows) == 1
    assert windows[0]["count"] == 1
    assert windows[0]["dur_mean"] == 1000.0


def test_rows_are_deduped_per_instance_batch(tmp_path):
    # One batch reported by two NPUs of the same instance (TP=2 decode, or
    # a PD prefill sender + shadow receiver) must count as ONE batch.
    lines = [_line(0, 1, 1000, sys_id=0), _line(0, 1, 1000, sys_id=4)]
    d = _write_batch(tmp_path, lines)
    windows = build_windows(d, window_s=2.0, roles={0: "decode"})
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
    windows = build_windows(d, window_s=2.0)  # no role argument
    roles = {w["instance_id"]: w["role"] for w in windows}
    assert roles == {0: "prefill", 1: "decode"}


def test_peer_ratio_is_leave_one_out(tmp_path):
    # Three decode instances; the third is 2x slower. A mean-including-self
    # reference would give 20/13.3 = 1.5; the LOO median gives 2.0.
    lines = [_line(0, 1, 10_000_000), _line(1, 1, 10_000_000),
             _line(2, 1, 20_000_000)]
    d = _write_batch(tmp_path, lines)
    roles = {0: "decode", 1: "decode", 2: "decode"}
    windows = build_windows(d, window_s=2.0, roles=roles)
    evaluate.add_peer_and_residual_features(
        [windows], evaluate.HealthyResidualModel())
    slow = [w for w in windows if w["instance_id"] == 2][0]
    assert slow["has_peer"] == 1.0
    assert abs(slow["dur_peer_ratio"] - 2.0) < 1e-9


def test_lone_instance_has_no_peer(tmp_path):
    # 1P1D: one decode instance, so peer features are undefined and must be
    # flagged, not silently reported as a meaningful 1.0.
    d = _write_batch(tmp_path, [_line(0, 1, 1000)])
    windows = build_windows(d, window_s=2.0, roles={0: "decode"})
    evaluate.add_peer_and_residual_features(
        [windows], evaluate.HealthyResidualModel())
    assert windows[0]["has_peer"] == 0.0
    assert windows[0]["dur_peer_ratio"] == 1.0


# ---------------------------------------------------------------------------
# labels from the audit (F3 / N1 / N7 / N8)
# ---------------------------------------------------------------------------
def _batch(inst, report_id, finish_s, dur_ns=10_000_000, sys_id=None):
    return "run_x,%d,%d,%d,%d,%d,%d,1,0,64,64,0,0,1,0\n" % (
        inst, report_id, inst if sys_id is None else sys_id,
        int((finish_s - 0.1) * 1e9), int(finish_s * 1e9), dur_ns)


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
    d = _write_batch(tmp_path, rows, audit=audit)
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
    d = _write_batch(tmp_path, rows, audit=audit)
    labels = label_windows(d, ENTRY, window_s=2.0, recovery_s=2.0)
    assert labels[(2.0, 1)]["kind"] == KIND_ROOT        # fault active
    assert labels[(4.0, 1)]["kind"] == KIND_RECOVERING  # inside recovery
    assert labels[(6.0, 1)]["kind"] == KIND_RECOVERING  # still inside
    assert labels[(8.0, 1)]["kind"] == KIND_UNAFFECTED  # after recovery


def test_neutral_mechanism_is_its_own_kind(tmp_path):
    rows = [_batch(1, 1, 1.0)]
    audit = [{"fault_id": "f0", "instance": 1, "batch_id": 0, "applied": False,
              "time_ns": 1_000_000_000}]
    d = _write_batch(tmp_path, rows, audit=audit)
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
    d = _write_batch(tmp_path, rows, audit=audit)
    labels = label_windows(d, entry, window_s=2.0)
    assert labels[(2.0, 0)]["kind"] == KIND_ROOT
    assert labels[(2.0, 0)]["label"] == "kv-transfer-network:0"
    assert labels[(2.0, 1)]["kind"] == KIND_DOWNSTREAM


def test_healthy_run_is_all_unaffected(tmp_path):
    rows = [_batch(0, 1, 1.0), _batch(1, 1, 1.5)]
    d = _write_batch(tmp_path, rows, audit=[])
    labels = label_windows(d, {"fault_present": False, "faults": []}, window_s=2.0)
    assert {lab["kind"] for lab in labels.values()} == {KIND_UNAFFECTED}


def test_labels_attach_to_matching_windows(tmp_path):
    rows = [_batch(0, 1, 1.0), _batch(1, 1, 1.2)]
    requests = ["0,0,1,10,20,10,0,0,0,0,0,1000,2000,0,1,0,1,1,1,1\n"]
    d = _write_batch(tmp_path, rows, requests=requests, audit=[])
    windows = build_windows(d, window_s=2.0)
    labelled = attach_labels(windows,
                            label_windows(d, {"faults": []}, window_s=2.0))
    assert len(labelled) == len(windows)
    assert all(w["label_kind"] == KIND_UNAFFECTED for w in labelled)
    # Roles came from the request telemetry, not from a single argument.
    roles = {w["instance_id"]: w["role"] for w in labelled}
    assert roles == {0: "prefill", 1: "decode"}


# ---------------------------------------------------------------------------
# manifest -> telemetry directory resolution
# ---------------------------------------------------------------------------
def _manifest_entry(**kw):
    base = {"pair_id": "pair_1", "workload_id": "w", "cluster_config_id": "c",
            "status": "ok", "fault_present": True, "faults": ENTRY["faults"],
            "regime": {}}
    base.update(kw)
    return json.dumps(base) + "\n"


def _variant(root, name, inst=1, audit_instance=None):
    d = root / "pair_1" / name
    d.mkdir(parents=True)
    audit = [{"fault_id": "f0",
              "instance": inst if audit_instance is None else audit_instance,
              "batch_id": 0, "applied": True, "time_ns": 1_000_000_000}]
    _write_batch(d, [_batch(inst, 1, 1.0)], audit=audit)


def _collect(root):
    return evaluate.collect_dataset(str(root), window_s=2.0,
                                    applied_share=evaluate.DEFAULT_APPLIED_SHARE,
                                    recovery_s=evaluate.DEFAULT_RECOVERY_S)


def test_multi_variant_pair_resolves_each_telemetry_dir(tmp_path):
    """A pair with several fault variants writes faulty0/1_telemetry.

    Deriving "faulty_telemetry" from `fault_present` skipped every faulty
    run of a multi-fault pair, which made a fully faulty matrix look like
    an all-unaffected dataset with zero positives.
    """
    root = tmp_path / "data"
    root.mkdir()
    healthy = root / "pair_1" / "healthy_telemetry"
    healthy.mkdir(parents=True)
    _write_batch(healthy, [_batch(1, 1, 1.0)])
    _variant(root, "faulty0_telemetry")
    _variant(root, "faulty1_telemetry")
    with open(root / "manifest.jsonl", "w") as f:
        f.write(_manifest_entry(run_id="h", fault_present=False, faults=[],
                                telemetry_dir="healthy_telemetry"))
        f.write(_manifest_entry(run_id="f0", telemetry_dir="faulty0_telemetry"))
        f.write(_manifest_entry(run_id="f1", telemetry_dir="faulty1_telemetry"))

    data = _collect(root)
    assert data["skipped_runs"] == []
    kinds = {(r["run_id"], r["label_kind"]) for r in data["rows"]}
    assert ("f0", KIND_ROOT) in kinds
    assert ("f1", KIND_ROOT) in kinds
    assert ("h", KIND_UNAFFECTED) in kinds


def test_legacy_manifest_without_telemetry_dir_still_resolves(tmp_path):
    """Bundles frozen before the field existed keep working."""
    root = tmp_path / "data"
    root.mkdir()
    _variant(root, "faulty_telemetry")
    with open(root / "manifest.jsonl", "w") as f:
        f.write(_manifest_entry(run_id="f0"))

    data = _collect(root)
    assert data["skipped_runs"] == []
    assert [r["label_kind"] for r in data["rows"]] == [KIND_ROOT]


def test_legacy_multi_variant_manifest_resolves_by_run_id(tmp_path):
    """In-flight bundles predate the field but their run_id names the twin.

    Falling back to `faulty0_telemetry` for every variant would score one
    run five times under five different run_ids.
    """
    root = tmp_path / "data"
    root.mkdir()
    for i in (0, 1, 2):
        _variant(root, "faulty%d_telemetry" % i, audit_instance=1)
    with open(root / "manifest.jsonl", "w") as f:
        for i in (0, 1, 2):
            f.write(_manifest_entry(run_id="pair_1_faulty%d" % i))

    data = _collect(root)
    assert data["skipped_runs"] == []
    resolved = {e["run_id"]: evaluate._telemetry_dir(str(root), e)
                for e in data["manifest"].values()}
    assert len(set(resolved.values())) == 3
    assert resolved["pair_1_faulty2"].endswith("faulty2_telemetry")


def test_timing_neutral_mechanism_is_excluded_not_scored(tmp_path):
    """A fault whose clocks are bit-identical to the healthy twin is not a target.

    kv_transfer_bandwidth at 0.8/0.67/0.5 gave identical total clocks in
    review_min: the audit says applied, the simulation says nothing
    happened. Scoring those windows as positives would count a guaranteed
    miss against every detector.
    """
    root = tmp_path / "data"
    root.mkdir()
    healthy = root / "pair_1" / "healthy_telemetry"
    healthy.mkdir(parents=True)
    _write_batch(healthy, [_batch(1, 1, 1.0)])
    _variant(root, "faulty0_telemetry")
    _variant(root, "faulty1_telemetry")
    with open(root / "manifest.jsonl", "w") as f:
        f.write(_manifest_entry(run_id="h", fault_present=False, faults=[],
                                total_clocks_ns=1000,
                                telemetry_dir="healthy_telemetry"))
        f.write(_manifest_entry(run_id="f0", total_clocks_ns=1000,
                                telemetry_dir="faulty0_telemetry"))
        f.write(_manifest_entry(run_id="f1", total_clocks_ns=1000,
                                telemetry_dir="faulty1_telemetry"))

    data = _collect(root)
    assert data["timing_neutral_faults"] == ["pair_1/f0"]
    kinds = {r["run_id"]: r["label_kind"] for r in data["rows"]}
    assert kinds["f0"] == KIND_NEUTRAL
    assert kinds["f1"] == KIND_NEUTRAL
    assert kinds["h"] == KIND_UNAFFECTED


def test_fault_that_moves_clocks_stays_a_positive(tmp_path):
    root = tmp_path / "data"
    root.mkdir()
    healthy = root / "pair_1" / "healthy_telemetry"
    healthy.mkdir(parents=True)
    _write_batch(healthy, [_batch(1, 1, 1.0)])
    _variant(root, "faulty0_telemetry")
    with open(root / "manifest.jsonl", "w") as f:
        f.write(_manifest_entry(run_id="h", fault_present=False, faults=[],
                                total_clocks_ns=1000,
                                telemetry_dir="healthy_telemetry"))
        f.write(_manifest_entry(run_id="f0", total_clocks_ns=1500,
                                telemetry_dir="faulty0_telemetry"))

    data = _collect(root)
    assert data["timing_neutral_faults"] == []
    kinds = {r["run_id"]: r["label_kind"] for r in data["rows"]}
    assert kinds["f0"] == KIND_ROOT


def test_dose_response_keeps_a_family_scoreable(tmp_path):
    """Different severities giving different clocks is a real signal."""
    root = tmp_path / "data"
    root.mkdir()
    healthy = root / "pair_1" / "healthy_telemetry"
    healthy.mkdir(parents=True)
    _write_batch(healthy, [_batch(1, 1, 1.0)])
    _variant(root, "faulty0_telemetry")
    _variant(root, "faulty1_telemetry")
    with open(root / "manifest.jsonl", "w") as f:
        f.write(_manifest_entry(run_id="h", fault_present=False, faults=[],
                                total_clocks_ns=1000,
                                telemetry_dir="healthy_telemetry"))
        f.write(_manifest_entry(run_id="f0", total_clocks_ns=1200,
                                telemetry_dir="faulty0_telemetry"))
        f.write(_manifest_entry(run_id="f1", total_clocks_ns=1500,
                                telemetry_dir="faulty1_telemetry"))

    data = _collect(root)
    assert data["timing_neutral_faults"] == []
    kinds = {r["run_id"]: r["label_kind"] for r in data["rows"]}
    assert kinds["f0"] == KIND_ROOT
    assert kinds["f1"] == KIND_ROOT


def test_missing_telemetry_dir_is_reported_not_silently_dropped(tmp_path):
    """A run that cannot be read must be visible in the report."""
    root = tmp_path / "data"
    root.mkdir()
    _variant(root, "faulty0_telemetry")
    with open(root / "manifest.jsonl", "w") as f:
        # No telemetry_dir and no legacy faulty_telemetry directory: this is
        # exactly the mismatch that hid the defect.
        f.write(_manifest_entry(run_id="f0"))

    data = _collect(root)
    assert data["rows"] == []
    assert data["skipped_runs"] == ["f0"]


# ---------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------
def test_wilson_interval_brackets_the_estimate():
    lo, hi = evaluate.wilson(5, 5)
    assert 0.0 <= lo <= 1.0 and lo < 1.0 and hi == 1.0
    lo, hi = evaluate.wilson(0, 8)
    assert lo == 0.0 and 0.0 < hi < 0.5


def test_mad_threshold_survives_one_benign_spike():
    # p99 over a handful of windows is the maximum; MAD is not.
    values = [100.0] * 14 + [10_000.0]
    assert evaluate.mad_threshold(values, k=4) < 10_000.0
    # Zero-spread healthy data still yields a threshold above the median.
    assert evaluate.mad_threshold([100.0] * 15, k=4) > 100.0


def test_cusum_accumulates_small_persistent_shift():
    rows = []
    for i in range(10):
        rows.append({"run_id": "r", "instance_id": 0, "role": "decode",
                     "window_end_s": float(i), "dur_residual": 0.0,
                     "label_kind": KIND_UNAFFECTED})
    # A persistent +3 sigma shift, below any single-window MAD threshold.
    for i in range(10, 14):
        rows.append({"run_id": "r", "instance_id": 0, "role": "decode",
                     "window_end_s": float(i), "dur_residual": 30.0,
                     "label_kind": KIND_ROOT})
    params = {"decode": (5.0, 25.0)}
    pred = evaluate.cusum_predict(rows, "dur_residual", params)
    assert pred[:10].sum() == 0
    assert pred[10:].sum() >= 1


def test_localisation_scores_root_cause_instance_only():
    rows = []
    for inst, residual in ((0, 1.0), (1, 500.0)):
        rows.append({"run_id": "r", "window_end_s": 2.0, "instance_id": inst,
                     "role": "decode", "dur_residual": residual,
                     "label_kind": KIND_ROOT if inst == 1 else KIND_DOWNSTREAM,
                     "label_instance_id": 1})
    loc = evaluate.localisation(rows, {"decode": 100.0})
    assert loc["windows"] == 1 and loc["scored"] == 1
    assert loc["top1"] == 1.0 and loc["coverage"] == 1.0
