"""Host-only tests for experiments/fault_detection/run_simulation.py.

Merged from test_day_workload.py (the NHPP day generator) plus coverage
for the pieces that module absorbed: the Poisson/bursty/sustained
workloads, the observable/ground-truth leak guard, the manifest readers,
audit consistency and regime measurement. No ASTRA-Sim binary needed.
"""

import csv
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from experiments.fault_detection import run_simulation as rs  # noqa: E402


def _cfg(**kw):
    cfg = dict(rs.DAY_DEFAULTS)
    cfg.update({"horizon_s": 2 * 3600.0, "base_sps": 2.0, "peak_hour": 1.0,
                "peak_height": 3.0, "peak_sigma_h": 0.25,
                "afternoon_tau_h": 3.0, "evening_frac": 0.4,
                "bursts": [], "seed": 7,
                "short_in": 32, "long_in": 128,
                "short_out": 16, "long_out": 64})
    cfg.update(kw)
    return cfg


# ---------------------------------------------------------------------------
# diurnal day workload
# ---------------------------------------------------------------------------
def test_day_schema_and_order():
    rows, _ = rs.generate_day_workload(_cfg())
    assert len(rows) > 0
    prev = -1
    for r in rows:
        assert set(r.keys()) == {"input_toks", "output_toks", "arrival_time_ns"}
        assert r["arrival_time_ns"] >= prev
        prev = r["arrival_time_ns"]
        assert 32 <= r["input_toks"] <= 128


def test_day_deterministic():
    assert (rs.generate_day_workload(_cfg())[0]
            == rs.generate_day_workload(_cfg())[0])
    assert (rs.generate_day_workload(_cfg(seed=8))[0]
            != rs.generate_day_workload(_cfg())[0])


def test_day_peak_shape():
    cfg = _cfg()
    r_peak = rs.day_rate_at(1.0 * 3600, cfg)
    r_early = rs.day_rate_at(600, cfg)
    assert r_peak > 2.0 * r_early  # peak_height=3 must dominate
    cfg_off = _cfg(peak_height=0.0)
    assert rs.day_rate_at(1.0 * 3600, cfg_off) < r_peak  # peak disabled


def test_day_bursts_add_load():
    a, _ = rs.generate_day_workload(_cfg())
    b, info = rs.generate_day_workload(
        _cfg(bursts=[{"dur_s": 60.0, "rate_mult": 2.0}] * 3))
    assert len(info["bursts"]) == 3
    assert len(b) > len(a)


def test_thinning_bound_covers_peak_plus_burst():
    """F4: the bound must be the supremum of the FULL intensity.

    The old bound was max(peak, burst) taken separately, which is not an
    upper bound when a burst overlaps the noon peak: rate/max > 1, every
    candidate is accepted and the process saturates at the bound.
    """
    cfg = _cfg(horizon_s=1.0 * 3600, peak_hour=0.5, peak_sigma_h=0.1)
    # Burst straddling the peak, exactly the case the old bound missed.
    cfg["bursts"] = [{"start_s": 0.5 * 3600 - 300, "dur_s": 600.0,
                      "rate_mult": 2.0}]
    grid_max = max(rs.intensity_at(t, cfg, cfg["bursts"])
                   for t in range(0, 3600))
    old_bound = max(cfg["base_sps"] * (1.0 + cfg["peak_height"]),
                    cfg["base_sps"] * 2.0)
    assert grid_max > old_bound            # the old bound was invalid here
    assert rs.max_intensity(cfg, cfg["bursts"]) >= grid_max - 1e-9


def test_realised_burst_volume_matches_analytic():
    """A burst must add ~rate_mult*base*dur requests, not a handful.

    Regression for the measured failure: a 600 s x2 burst at the peak
    added 22 requests where the model implies ~1200.
    """
    cfg = _cfg(horizon_s=3.0 * 3600, peak_hour=1.5, peak_sigma_h=0.3)
    start, dur = 1.5 * 3600 - 900, 1800.0
    cfg["bursts"] = [{"start_s": start, "dur_s": dur, "rate_mult": 2.0}]
    rows, _info = rs.generate_day_workload(cfg)
    inside = sum(1 for r in rows
                 if start <= r["arrival_time_ns"] / 1e9 < start + dur)
    expected = sum(rs.intensity_at(start + i, cfg, cfg["bursts"])
                   for i in range(int(dur)))
    assert abs(inside - expected) / expected < 0.05, (inside, expected)


def test_day_boundary_is_documented_not_periodic():
    """F5: the profile is a ONE-SHOT day; pin the discontinuity.

    If this ever becomes periodic, the docstring and report must change
    with it -- the test exists so the property cannot drift silently.
    """
    cfg = _cfg(horizon_s=24.0 * 3600, peak_hour=12.0, peak_sigma_h=1.5)
    assert rs.day_rate_at(23.99 * 3600, cfg) < 0.1 * rs.day_rate_at(0.1 * 3600, cfg)


# ---------------------------------------------------------------------------
# poisson / bursty / sustained workloads
# ---------------------------------------------------------------------------
def test_poisson_workload_is_seeded_and_ordered():
    a = rs.generate_workload(num=20, rate_sps=4.0, seed=5)
    b = rs.generate_workload(num=20, rate_sps=4.0, seed=5)
    assert a == b
    assert len(a) == 20
    times = [r["arrival_time_ns"] for r in a]
    assert times == sorted(times)
    assert all(32 <= r["input_toks"] <= 512 for r in a)


def test_sustained_workload_has_exact_gaps():
    rows = rs.generate_workload(num=4, rate_sps=2.0, seed=1, pattern="sustained")
    gaps = [rows[i]["arrival_time_ns"] - rows[i - 1]["arrival_time_ns"]
            for i in range(1, 4)]
    assert gaps == [500_000_000] * 3


def test_bursty_workload_keeps_exact_count():
    rows = rs.generate_workload(num=17, rate_sps=8.0, seed=2,
                                pattern="bursty", on_s=1.0, off_s=2.0)
    assert len(rows) == 17


def test_write_workload_roundtrip(tmp_path):
    path = tmp_path / "w.jsonl"
    rows = rs.generate_workload(num=3, rate_sps=1.0, seed=3)
    assert rs.write_workload(str(path), rows) == 3
    loaded = [json.loads(line) for line in path.read_text().splitlines()]
    assert loaded == rows


# ---------------------------------------------------------------------------
# observable / ground-truth separation (N6)
# ---------------------------------------------------------------------------
BATCH_HEADER = ("run_id,instance_id,batch_id,sys,batch_time_ns,finish_ns,"
                "duration_ns,num_prefill,num_decode,total_len,kv_len,"
                "pd_kv_send_tokens,n_waiting,n_running,com_cycle\n")


def _write_telemetry(tmp_path, lines, header=BATCH_HEADER):
    with open(tmp_path / "batch_telemetry.csv", "w") as f:
        f.write(header)
        f.writelines(lines)
    return str(tmp_path)


def test_observable_clean_accepts_opaque_run_id(tmp_path):
    d = _write_telemetry(tmp_path, ["run_ab12cd34ef56,0,1,0,0,1000,1000,1,0,64,64,0,0,1,0\n"])
    rs.check_observable_clean(d)  # must not raise


def test_observable_clean_rejects_ground_truth_column(tmp_path):
    header = BATCH_HEADER.replace("run_id", "run_id,fault_present")
    d = _write_telemetry(tmp_path, ["run_ab,1,0,1,0,0,1000,1000,1,0,64,64,0,0,1,0\n"],
                         header=header)
    with pytest.raises(ValueError) as exc:
        rs.check_observable_clean(d)
    assert "fault_present" in str(exc.value)


@pytest.mark.parametrize("token", ["fault", "healthy", "slow", "stall",
                                   "degrad", "throttle", "neutral"])
def test_observable_clean_rejects_leaky_values(tmp_path, token):
    """N6: a run_id that spells out the label leaks as badly as a column."""
    d = _write_telemetry(tmp_path, ["decode_%s_2x,0,1,0,0,1000,1000,1,0,64,64,0,0,1,0\n"
                                    % token])
    with pytest.raises(ValueError) as exc:
        rs.check_observable_clean(d)
    assert "opaque run id" in str(exc.value)


def test_manifest_only_columns_cover_every_label_column():
    for column in ("fault_present", "fault_id", "failure_class",
                   "successfully_applied", "pair_id", "expected_label"):
        assert column in rs.MANIFEST_ONLY_COLUMNS


# ---------------------------------------------------------------------------
# manifest + audit consistency
# ---------------------------------------------------------------------------
FAULT = {
    "fault_id": "f0",
    "mode": "compute_multiplicative",
    "target": {"role": "decode", "instance_id": 1},
    "factor": 2.0,
    "schedule": {"start_s": 1.0, "end_s": 2.0, "profile": "step"},
}


def test_manifest_roundtrip(tmp_path):
    path = tmp_path / "manifest.jsonl"
    entries = [{"run_id": "a", "status": "ok", "fault_present": False},
               {"run_id": "b", "status": "failed", "fault_present": True}]
    with open(path, "w") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")
    assert rs.load_manifest(str(path)) == entries


def test_expected_label_resolves_role_and_transfer():
    assert rs.expected_label_for_fault(FAULT) == "decode"
    transfer = dict(FAULT, mode="kv_transfer_bandwidth")
    assert rs.expected_label_for_fault(transfer) == "kv-transfer-network"
    stall = dict(FAULT, mode="worker_stall")
    assert rs.expected_label_for_fault(stall) == "worker-unavailable"


def test_audit_consistency_requires_applied_record_inside_window():
    ok = [{"fault_id": "f0", "time_ns": 1_500_000_000, "applied": True}]
    entry = {"fault_present": True, "faults": [FAULT]}
    assert rs.check_audit_consistency(entry, ok)[0] is True
    # An applied record OUTSIDE the schedule window does not count.
    outside = [{"fault_id": "f0", "time_ns": 5_000_000_000, "applied": True}]
    assert rs.check_audit_consistency(entry, outside)[0] is False
    # A visited-but-neutral window is a finding, not a success.
    neutral = [{"fault_id": "f0", "time_ns": 1_500_000_000, "applied": False}]
    ok_flag, reason = rs.check_audit_consistency(entry, neutral)
    assert ok_flag is False and "applied=false" in reason


def test_healthy_run_must_have_empty_audit():
    assert rs.check_audit_consistency({"fault_present": False, "faults": []}, [])[0]
    assert rs.check_audit_consistency(
        {"fault_present": False, "faults": []},
        [{"fault_id": "f0", "time_ns": 1, "applied": True}])[0] is False


def test_total_clocks_parses_log():
    log = "some noise\nTotal clocks (ns): 1661784496\nmore\n"
    assert rs.total_clocks_from_log(log) == 1661784496
    assert rs.total_clocks_from_log("no clock line") == -1


# ---------------------------------------------------------------------------
# regime: a detection number without its load is not interpretable
# ---------------------------------------------------------------------------
REQUEST_HEADER = ("request_id,prefill_instance_id,decode_instance_id,"
                  "input_toks,output_toks,arrival_ns,first_scheduled_ns,"
                  "prefill_done_ns,decode_admit_ns,first_decode_done_ns,"
                  "ttft_ns,latency_ns,queuing_delay_ns,tpot,itl_count,"
                  "itl_mean_ns,itl_p50_ns,itl_p99_ns,itl_max_ns\n")


def test_regime_from_telemetry_measures_utilisation(tmp_path):
    # Two instances, one batch each. Batch 0 runs 0->10s, batch 1 runs
    # 10->20s, so the run spans 20 s and each instance is busy 10 s (0.5).
    lines = ["run_x,0,1,0,0,10000000000,10000000000,1,0,64,64,0,0,1,0\n",
             "run_x,1,1,1,10000000000,20000000000,10000000000,1,0,64,64,0,0,1,0\n"]
    with open(tmp_path / "batch_telemetry.csv", "w") as f:
        f.write(BATCH_HEADER)
        f.writelines(lines)
    with open(tmp_path / "request_telemetry.csv", "w") as f:
        f.write(REQUEST_HEADER)
        f.write("0,0,1,10,20,0,0,0,0,0,1000,2000,0,5000000,1,0,0,0,0\n")
    regime = rs.regime_from_telemetry(str(tmp_path))
    assert regime["span_s"] == pytest.approx(20.0)
    assert regime["u_prefill"] == pytest.approx(0.5)
    assert regime["u_decode"] == pytest.approx(0.5)
    assert regime["n_requests"] == 1
    assert regime["tpot_p50_ms"] == pytest.approx(5.0)


def test_regime_dedupes_duplicate_npu_reports(tmp_path):
    # A TP=2 instance reports each batch twice (sys 0 and sys 1). Busy
    # time must not double-count: two 2 s batches inside a 10 s span is
    # 0.4, not 0.8.
    lines = ["run_x,1,1,0,0,2000000000,2000000000,1,0,64,64,0,0,1,0\n",
             "run_x,1,1,1,0,2000000000,2000000000,1,0,64,64,0,0,1,0\n",
             "run_x,1,2,0,8000000000,10000000000,2000000000,1,0,64,64,0,0,1,0\n",
             "run_x,1,2,1,8000000000,10000000000,2000000000,1,0,64,64,0,0,1,0\n"]
    with open(tmp_path / "batch_telemetry.csv", "w") as f:
        f.write(BATCH_HEADER)
        f.writelines(lines)
    with open(tmp_path / "request_telemetry.csv", "w") as f:
        f.write(REQUEST_HEADER)
        f.write("0,0,1,10,20,0,0,0,0,0,1000,2000,0,5000000,1,0,0,0,0\n")
    regime = rs.regime_from_telemetry(str(tmp_path))
    assert regime["span_s"] == pytest.approx(10.0)
    assert regime["u_decode"] == pytest.approx(0.4)


def test_regime_handles_missing_telemetry(tmp_path):
    regime = rs.regime_from_telemetry(str(tmp_path))
    assert regime["u_decode"] == -1.0 and regime["n_requests"] == 0


# ---------------------------------------------------------------------------
# simulator invocation (no binary needed: the command is inspectable)
# ---------------------------------------------------------------------------
def test_run_simulator_reports_timeout(monkeypatch, tmp_path):
    import subprocess

    def _boom(*_a, **_kw):
        raise subprocess.TimeoutExpired(cmd="x", timeout=1)

    monkeypatch.setattr(rs.subprocess, "run", _boom)
    workload = tmp_path / "w.jsonl"
    workload.write_text(json.dumps({"input_toks": 8, "output_toks": 8,
                                    "arrival_time_ns": 0}) + "\n")
    result = rs.run_simulator(str(tmp_path), "cluster.json", str(workload),
                              str(tmp_path / "out.csv"), str(tmp_path / "tel"),
                              "run_x", timeout=1)
    assert result["status"] == "failed"
    assert result["reason"] == "timeout"
    assert result["clocks"] == -1


def test_run_simulator_passes_jitter_only_when_enabled(monkeypatch, tmp_path):
    seen = {}

    class _Proc:
        returncode = 0
        stdout = "Total clocks (ns): 42\n"
        stderr = ""

    def _capture(cmd, **_kw):
        seen["cmd"] = cmd
        return _Proc()

    monkeypatch.setattr(rs.subprocess, "run", _capture)
    workload = tmp_path / "w.jsonl"
    workload.write_text(json.dumps({"input_toks": 8, "output_toks": 8,
                                    "arrival_time_ns": 0}) + "\n")
    kwargs = dict(repo_root=str(tmp_path), cluster="c.json",
                  workload_path=str(workload),
                  out_csv=str(tmp_path / "o.csv"),
                  telemetry_dir=str(tmp_path / "tel"), run_id="run_x")
    rs.run_simulator(**kwargs)
    assert "--timing-jitter" not in seen["cmd"]
    rs.run_simulator(timing_jitter=0.05, **kwargs)
    assert "--timing-jitter" in seen["cmd"]


def test_run_simulator_passes_throttle_config(monkeypatch, tmp_path):
    seen = {}

    class _Proc:
        returncode = 0
        stdout = ""
        stderr = ""

    def _capture(cmd, **_kw):
        seen["cmd"] = cmd
        return _Proc()

    monkeypatch.setattr(rs.subprocess, "run", _capture)
    workload = tmp_path / "w.jsonl"
    workload.write_text(json.dumps({"input_toks": 8, "output_toks": 8,
                                    "arrival_time_ns": 0}) + "\n")
    rs.run_simulator(str(tmp_path), "c.json", str(workload),
                     str(tmp_path / "o.csv"), str(tmp_path / "tel"), "run_x",
                     throttle_path=str(tmp_path / "thr.json"))
    assert "--throttle-config" in seen["cmd"]


# ---------------------------------------------------------------------------
# paired runner: workload construction and throttle targeting
# ---------------------------------------------------------------------------
def test_build_workload_keeps_the_requested_pattern(tmp_path):
    """Regression: popping `pattern` turned every bursty trace into Poisson.

    The smoke re-run caught it by clock mismatch (bursty == heavy). A
    pattern-blind builder is invisible in the manifest, so pin it here.
    """
    from experiments.fault_detection.inject_faults import _build_workload

    job = {"workload": {"pattern": "bursty", "num": 12, "rate_sps": 8.0,
                        "on_s": 1.0, "off_s": 2.0, "short_in": 32,
                        "long_in": 512, "short_out": 32, "long_out": 256}}
    out = tmp_path / "w.jsonl"
    _build_workload(job, seed=13, path=str(out))
    written = [json.loads(line) for line in out.read_text().splitlines()]
    expected = rs.generate_workload(
        pattern="bursty", num=12, rate_sps=8.0, seed=13, on_s=1.0, off_s=2.0,
        short_in=32, long_in=512, short_out=32, long_out=256)
    assert written == expected
    poisson = rs.generate_workload(
        pattern="poisson", num=12, rate_sps=8.0, seed=13,
        short_in=32, long_in=512, short_out=32, long_out=256)
    assert written != poisson


def test_throttle_gpus_skip_prefill_shadow_range():
    """Decode NPUs of the 2P2D tp=1 config are 4 and 5.

    Each prefill instance reserves 1 physical NPU + 1 shadow receiver, so
    a naive tp*pp counter would throttle the wrong hardware.
    """
    from experiments.fault_detection.inject_faults import throttle_gpus_for_instance

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    cluster = "configs/cluster/pd_2p2d_small.json"
    assert throttle_gpus_for_instance(cluster, repo_root, 0) == [0]
    assert throttle_gpus_for_instance(cluster, repo_root, 1) == [2]
    assert throttle_gpus_for_instance(cluster, repo_root, 2) == [4]
    assert throttle_gpus_for_instance(cluster, repo_root, 3) == [5]


def test_probe_run_ids_are_disjoint_and_path_safe():
    """Two calibrations must not share astra-sim/inputs/runs/<run_id>.

    Measured failure: two concurrent calibrations both used run_id
    "probe_01", so each rewrote the other's workload inside the shared
    input tree. The run id has to carry the out-root, and it has to match
    the simulator's allowed charset.
    """
    a = rs.probe_run_id("experiments/out/pilot_2p2d_u40", "probe_01")
    b = rs.probe_run_id("experiments/out/pilot_2p2d_u75", "probe_01")
    assert a != b
    assert rs.probe_run_id("experiments/out/pilot_2p2d_u40", "probe_00") != a
    for value in (a, b):
        assert rs.re.match(r"^[A-Za-z0-9_.-]+$", value), value
        assert "/" not in value and "\\" not in value


def test_calibrate_stays_inside_its_explicit_bracket(monkeypatch, tmp_path):
    """lambda_lo/lambda_hi must bound every probe.

    Near saturation U is flat in lambda, so the default two-decade bracket
    spends probes returning U=1.0. Narrowing the bracket is only safe if
    the search actually honours it -- that is what this pins, using a fake
    simulator whose regime is a known function of the trace it was handed
    (U = lambda/2, so U = 0.5 at lambda = 1.0).
    """
    probed = []

    def fake_run_simulator(repo_root, cluster, workload_path, out_csv,
                           telemetry_dir, run_id, **_kw):
        rows = [json.loads(line) for line in open(workload_path)]
        span_s = (rows[-1]["arrival_time_ns"] - rows[0]["arrival_time_ns"]) / 1e9
        probed.append((len(rows) - 1) / span_s)
        return {"status": "ok", "reason": "", "clocks": 1, "log": "",
                "returncode": 0, "cmd": []}

    def fake_regime(_telemetry_dir):
        lam = probed[-1]
        return {"lambda_sps": lam, "u_prefill": 0.0,
                "u_decode": min(1.0, 0.5 * lam), "tpot_p50_ms": 1.0,
                "span_s": 1.0, "n_requests": 20}

    monkeypatch.setattr(rs, "run_simulator", fake_run_simulator)
    monkeypatch.setattr(rs, "regime_from_telemetry", fake_regime)

    record = rs.calibrate_load(str(tmp_path), str(tmp_path / "out"), "c.json",
                               target_u=0.5, num_reqs=20, iters=4, tol=0.05,
                               lambda_lo=0.5, lambda_hi=4.0)
    assert probed, "calibration must probe at least once"
    for lam in probed:
        assert 0.5 - 1e-9 <= lam <= 4.0 + 1e-9, lam
    assert abs(record["chosen"]["u_decode"] - 0.5) <= 0.05
    assert record["chosen"]["tag"].startswith("probe_")


# ---------------------------------------------------------------------------
# manifest durability: a pair is many runs, so flush per RUN
# ---------------------------------------------------------------------------
def test_manifest_is_flushed_per_run_not_per_pair(monkeypatch, tmp_path):
    """One long pair must not hold back the manifest lines of finished runs.

    A review_min pair is up to 12 simulator runs (20-40 min). Writing the
    manifest only when the whole pair returns means an interrupted matrix
    re-simulates runs that already succeeded -- measured cost, not a
    hypothetical.
    """
    import threading

    from experiments.fault_detection import inject_faults as inj

    def fake_run_simulator(repo_root, cluster, workload_path, out_csv,
                           telemetry_dir, run_id, **_kw):
        os.makedirs(telemetry_dir, exist_ok=True)
        with open(os.path.join(telemetry_dir, "batch_telemetry.csv"), "w") as f:
            f.write("run_id,instance_id,batch_id,sys,batch_time_ns,finish_ns,"
                    "duration_ns,num_prefill,num_decode,total_len,kv_len,"
                    "pd_kv_send_tokens,n_waiting,n_running,com_cycle\n")
            f.write("%s,0,1,0,0,1000000000,1000000000,1,0,64,64,0,0,1,0\n"
                    % run_id)
        return {"status": "ok", "reason": "", "clocks": 1, "log": "",
                "returncode": 0, "cmd": []}

    monkeypatch.setattr(inj, "run_simulator", fake_run_simulator)
    monkeypatch.setattr(inj, "git_commit", lambda _root: "deadbee")

    manifest_path = tmp_path / "manifest.jsonl"
    (tmp_path / "c.json").write_text(json.dumps({"nodes": [{"instances": [
        {"pd_type": "prefill", "tp_size": 1}, {"pd_type": "decode", "tp_size": 1}]}]}))
    job = {"workload_id": "w", "cluster_id": "c", "cluster": "c.json",
           "workload": {"pattern": "poisson", "num": 2, "rate_sps": 1.0,
                        "short_in": 32, "long_in": 64, "short_out": 8,
                        "long_out": 16}}
    fault = {"fault_id": "f", "mode": "compute_multiplicative",
             "target": {"role": "decode", "instance_id": 1}, "factor": 2.0,
             "schedule": {"start_s": 0.0, "end_s": 1.0, "profile": "step"}}
    with open(manifest_path, "a") as handle:
        inj.run_pair(str(tmp_path), str(tmp_path / "out"), "p1", job, [fault],
                     seed=1, timeout=10, keep_inputs=False, commit="deadbee",
                     profile_name="t", manifest_file=handle,
                     manifest_lock=threading.Lock())
    lines = [json.loads(line) for line in manifest_path.read_text().splitlines()]
    assert [entry["run_id"] for entry in lines] == ["p1_healthy", "p1_faulty"]
    assert lines[0]["fault_present"] is False
