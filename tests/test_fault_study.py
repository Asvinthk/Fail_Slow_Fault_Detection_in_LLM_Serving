"""Host-only tests for serving/core/fault_study.py (no simulator binary).

Merged from the three modules that fault_study.py consolidates:
test_fault_injection.py, test_gpu_throttle.py and test_timing_jitter.py.
Every original assertion is preserved; the tick tests now call the
module-level `tick_throttle` instead of a private `__main__` helper.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from serving.core.fault_study import (  # noqa: E402
    GpuThrottleState,
    ThrottleManager,
    apply_throttle_factor,
    apply_timing_jitter,
    apply_to_trace_rows,
    audit_stall_windows,
    audit_handoff_hold,
    fault_matches,
    is_stalled,
    jitter_factor,
    note_stall_skip,
    select_handoff_hold,
    severity_weight,
    tick_throttle,
    validate_fault_config,
    validate_throttle_config,
)

from serving.core.fault_study import GpuThrottleState


def make_row(layer="attention_0", comp=1000, comm=0):
    return [layer, str(comp), "LOCAL", "10", "LOCAL", "10",
            "LOCAL", "10", "NONE", str(comm), "NONE"]


STEP = {"start_s": 0.0, "end_s": 10.0, "profile": "step"}


# ---------------------------------------------------------------------------
# severity schedule
# ---------------------------------------------------------------------------
def test_step_weight():
    schedule = {"start_s": 1.0, "end_s": 2.0, "profile": "step"}
    assert severity_weight(schedule, 0.5) == 0.0
    assert severity_weight(schedule, 1.5) == 1.0
    assert severity_weight(schedule, 2.5) == 0.0


def test_intermittent_and_ramp():
    inter = {"start_s": 0.0, "end_s": 10.0, "profile": "intermittent",
             "period_s": 2.0, "duty": 0.5}
    assert severity_weight(inter, 0.5) == 1.0
    assert severity_weight(inter, 1.5) == 0.0
    ramp = {"start_s": 0.0, "end_s": 4.0, "profile": "ramp_up", "ramp_s": 4.0}
    assert severity_weight(ramp, 2.0) == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# compute faults
# ---------------------------------------------------------------------------
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
    faults = [{"fault_id": "f1", "target": {"role": "decode", "instance_id": 1},
               "injection_layer": "mechanism", "mode": "compute_multiplicative",
               "factor": 2.0, "schedule": STEP}]
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
    faults = [{"fault_id": "f1", "target": {"role": "decode", "instance_id": 1},
               "injection_layer": "mechanism", "mode": "compute_multiplicative",
               "factor": 2.0, "schedule": STEP}]
    rows = [tuple(make_row(comp=1000))]
    audits = apply_to_trace_rows(faults, rows, "decode", 1, 5.0)
    assert rows[0][1] == "2000"
    assert audits[0]["applied"] is True


def test_additive_delta():
    faults = [{"fault_id": "f1", "target": {"role": "prefill", "instance_id": 0},
               "injection_layer": "mechanism", "mode": "compute_additive",
               "delta_ns": 500, "schedule": STEP}]
    rows = [make_row(comp=1000)]
    apply_to_trace_rows(faults, rows, "prefill", 0, 5.0)
    assert rows[0][1] == "1500"


def test_additive_applies_once_per_iteration_not_per_layer():
    faults = [{"fault_id": "f1", "target": {"role": "decode", "instance_id": 0},
               "injection_layer": "mechanism", "mode": "compute_additive",
               "delta_ns": 500, "schedule": STEP}]
    rows = [make_row(layer="attention_0", comp=1000),
            make_row(layer="mlp_0", comp=2000),
            make_row(layer="sampler_0", comp=3000)]
    audits = apply_to_trace_rows(faults, rows, "decode", 0, 5.0)
    assert rows[0][1] == "1000"
    assert rows[1][1] == "2000"
    assert rows[2][1] == "3500"
    assert audits[0]["applied_comp_sum"] - audits[0]["nominal_comp_sum"] == 500


def test_moe_and_pim_marker_rows_untouched():
    faults = [{"fault_id": "f1", "target": {"role": "decode", "instance_id": 0},
               "injection_layer": "mechanism", "mode": "compute_multiplicative",
               "factor": 2.0, "schedule": STEP}]
    marker = ["EXPERT 0", "ALLTOALL:0,1"]
    rows = [make_row(comp=1000), marker]
    apply_to_trace_rows(faults, rows, "decode", 0, 5.0)
    assert rows[0][1] == "2000"
    assert rows[1] == marker


# ---------------------------------------------------------------------------
# transfer faults
# ---------------------------------------------------------------------------
def test_kv_bandwidth_inflates_comm_size():
    faults = [{"fault_id": "f1", "target": {"role": "transfer", "instance_id": 0},
               "injection_layer": "mechanism", "mode": "kv_transfer_bandwidth",
               "bandwidth_fraction": 0.5, "schedule": STEP}]
    rows = [make_row(layer="qkv_proj_0", comp=1000, comm=800)]
    apply_to_trace_rows(faults, rows, "prefill", 0, 5.0)
    assert rows[0][9] == "1600"


def test_kv_bandwidth_respects_partial_weight():
    schedule = {"start_s": 0.0, "end_s": 4.0, "profile": "ramp_up", "ramp_s": 4.0}
    faults = [{"fault_id": "f1", "target": {"role": "transfer", "instance_id": 0},
               "injection_layer": "mechanism", "mode": "kv_transfer_bandwidth",
               "bandwidth_fraction": 0.5, "schedule": schedule}]
    rows = [make_row(layer="qkv_proj_0", comp=1000, comm=800)]
    apply_to_trace_rows(faults, rows, "prefill", 0, 2.0)
    # w(2s) = 0.5 -> effective inflation 1 + 0.5*(2-1) = 1.5x
    assert rows[0][9] == "1200"


def test_network_degradation_scales_collective_comm_only():
    faults = [{"fault_id": "n1", "target": {"role": "network", "instance_id": 0},
               "injection_layer": "mechanism", "mode": "network_degradation",
               "factor": 2.0, "schedule": STEP}]

    def comm_row(layer, comp, comm):
        row = make_row(layer=layer, comp=comp, comm=comm)
        row[8] = "ALLREDUCE"
        return row

    rows = [comm_row("o_proj_0", 1000, 800), make_row(comp=1000, comm=0)]
    audits = apply_to_trace_rows(faults, rows, "decode", 0, 5.0)
    assert rows[0][1] == "1000"  # comp untouched by network faults
    assert rows[0][9] == "1600"
    assert rows[1][9] == "0"
    assert audits[0]["applied_comm_sum"] == 1600


def test_network_target_matches_instance_regardless_of_role():
    assert fault_matches(
        {"target": {"role": "network", "instance_id": 1}}, "prefill", 1) is True
    assert fault_matches(
        {"target": {"role": "network", "instance_id": 1}}, "decode", 0) is False


def test_neutral_batches_are_not_labeled_applied():
    # tp=1 dense trace: no collective rows, so a network fault visits
    # batches but changes nothing -> applied must be False, never silent.
    faults = [{"fault_id": "n1", "target": {"role": "network", "instance_id": 0},
               "injection_layer": "mechanism", "mode": "network_degradation",
               "factor": 1.5, "schedule": STEP}]
    rows = [make_row(comp=1000, comm=0)]
    audits = apply_to_trace_rows(faults, rows, "decode", 0, 5.0)
    assert len(audits) == 1
    assert audits[0]["applied"] is False
    assert "note" in audits[0]


def test_handoff_hold_selects_strongest_active_fault():
    faults = [
        {"fault_id": "h1", "target": {"role": "transfer", "instance_id": 0},
         "injection_layer": "mechanism", "mode": "kv_handoff_delay",
         "delay_ns": 50_000_000, "schedule": STEP},
        {"fault_id": "h2", "target": {"role": "transfer", "instance_id": 0},
         "injection_layer": "mechanism", "mode": "kv_handoff_delay",
         "delay_ns": 100_000_000, "schedule": STEP},
    ]
    delay, fault = select_handoff_hold(faults, "prefill", 0, 5.0)
    assert delay == 100_000_000
    assert fault["fault_id"] == "h2"
    # Outside the window: no hold at all.
    assert select_handoff_hold(faults, "prefill", 0, 50.0) == (0, None)
    row = audit_handoff_hold(fault, 0, 5_000_000_000, delay, 3, 1.0)
    assert row["type"] == "handoff_hold" and row["applied"] is True
    assert row["delay_ns"] == delay and row["count"] == 3


# ---------------------------------------------------------------------------
# stall
# ---------------------------------------------------------------------------
def test_stall_and_validation():
    faults = [{"fault_id": "s1", "target": {"role": "decode", "instance_id": 0},
               "injection_layer": "mechanism", "mode": "worker_stall",
               "schedule": STEP}]
    assert is_stalled(faults, "decode", 0, 5.0) is True
    assert is_stalled(faults, "decode", 1, 5.0) is False
    with pytest.raises(ValueError):
        validate_fault_config({"faults": [{
            "fault_id": "x", "target": {"role": "decode", "instance_id": 0},
            "injection_layer": "mechanism", "mode": "nope",
            "schedule": STEP}]})
    with pytest.raises(ValueError):
        validate_fault_config({"faults": [], "bogus_key": 1})


def test_stall_skip_counting_and_summary():
    faults = [{"fault_id": "s1", "target": {"role": "decode", "instance_id": 0},
               "injection_layer": "mechanism", "mode": "worker_stall",
               "schedule": {"start_s": 2.0, "end_s": 4.0, "profile": "step"}}]
    skips = {}
    note_stall_skip(skips, faults, "decode", 0, 3.0)
    note_stall_skip(skips, faults, "decode", 0, 3.1)
    note_stall_skip(skips, faults, "decode", 0, 9.0)   # outside window
    note_stall_skip(skips, faults, "decode", 1, 3.0)   # other instance
    assert skips == {("s1", 0): 2}
    rows = audit_stall_windows(skips, faults)
    assert len(rows) == 1
    assert rows[0]["type"] == "worker_stall_window"
    assert rows[0]["skipped_schedules"] == 2
    assert rows[0]["applied"] is True
    # Absence of work is the evidence: no skips means not applied.
    assert audit_stall_windows({("s1", 0): 0}, faults)[0]["applied"] is False


# ---------------------------------------------------------------------------
# router interaction (unchanged upstream behaviour, kept as regression)
# ---------------------------------------------------------------------------
class _StubRouter:
    def __init__(self, nxt=None):
        self._nxt = nxt

    def get_next_pending_arrival(self):
        return self._nxt


class _FakeSched:
    def __init__(self, instance_id, pd_type):
        self.instance_id = instance_id
        self.pd_type = pd_type
        self.waiting = []
        self.running = []
        self.admitted = []

    def add_decode(self, req):
        self.admitted.append(req)


def _make_router(n_decode=1):
    from serving.core.router import Router
    decodes = [_FakeSched(i + 100, "decode") for i in range(n_decode)]
    prefill = [_FakeSched(0, "prefill")]
    router = Router(num_instances=n_decode + 1,
                    schedulers=prefill + decodes, req_num=0,
                    routing_policy="RR", seed=1)
    return router, decodes


def test_stalled_decode_holds_instead_of_dropping():
    router, decodes = _make_router(n_decode=1)
    req = object()
    router.transfer_prefill_request([req], skip_decode_ids={100})
    assert decodes[0].admitted == []
    assert router.has_held_decode() is True
    # Still stalled: stays held.
    assert router.retry_held_decode(skip_decode_ids={100}) == []
    assert router.has_held_decode() is True
    # Recovered: admitted exactly once.
    admitted = router.retry_held_decode(skip_decode_ids=set())
    assert admitted == [req]
    assert decodes[0].admitted == [req]
    assert router.has_held_decode() is False


def test_transfer_prefers_available_decode():
    router, decodes = _make_router(n_decode=2)
    req = object()
    router.transfer_prefill_request([req], skip_decode_ids={100})
    assert router.has_held_decode() is False
    assert decodes[1].admitted == [req]
    assert decodes[0].admitted == []


def test_pass_response_unchanged_without_faults():
    from serving.__main__ import _pass_response
    router = _StubRouter(nxt=5_000_000_000)
    assert _pass_response(router, 1_000_000_000) == "pass 5000000000"
    assert _pass_response(_StubRouter(), 1_000_000_000) == "pass"


def test_pass_response_wakes_at_fault_boundaries_and_holds():
    from serving.__main__ import _pass_response
    router = _StubRouter(nxt=None)
    faults = [{"fault_id": "s1", "target": {"role": "decode", "instance_id": 0},
               "injection_layer": "mechanism", "mode": "worker_stall",
               "schedule": {"start_s": 2.0, "end_s": 4.0, "profile": "step"}}]
    # Fault start (2e9 ns) is the earliest wake-up.
    assert _pass_response(router, 1_000_000_000,
                          fault_faults=faults, held=[]) == "pass 2000000000"
    # A handoff release earlier than the fault end wins.
    held = [(2_500_000_000, object())]
    assert _pass_response(router, 1_000_000_000,
                          fault_faults=faults, held=held) == "pass 2000000000"
    # A past handoff release never drags the deadline backwards;
    # the next wake-up is the fault end.
    assert _pass_response(router, 3_000_000_000,
                          fault_faults=faults, held=held) == "pass 4000000000"


# ---------------------------------------------------------------------------
# throttle
# ---------------------------------------------------------------------------
def test_disabled_manager_is_neutral():
    m = ThrottleManager(None)
    assert not m.enabled()
    assert m.factor_for(0) == 1.0
    assert m.update_window(0, 1.0, 60.0, 60.0) is None


def test_accumulation_and_decay():
    st = GpuThrottleState(gpu_id=0, tau_s=600.0, th_on=0.75, th_off=0.60)
    for _ in range(20):  # sustained full load
        st.update(1.0, 60.0, 0.0)
    assert st.load > 0.75 and st.throttled
    assert st.factor() == 1.5
    for _ in range(40):  # idle recovery
        st.update(0.0, 60.0, 0.0)
    assert st.load < 0.60 and not st.throttled
    assert st.factor() == 1.0


def test_hysteresis_no_flap():
    st = GpuThrottleState(gpu_id=0, tau_s=60.0, th_on=0.75, th_off=0.60)
    st.load = 0.70  # between thresholds, starts unthrottled
    assert st.update(0.70, 60.0, 1.0) is None
    assert not st.throttled
    st.load = 0.76
    assert st.update(0.76, 1.0, 2.0) == (2.0, True)
    st.load = 0.65  # above th_off: stays throttled
    assert st.update(0.65, 1.0, 3.0) is None
    assert st.throttled


def test_m_min_clamp():
    st = GpuThrottleState(gpu_id=0, slowdown_factor=1.5, m_min=0.8)
    st.throttled = True
    assert st.factor() == min(1.5, 1.0 / 0.8)


def test_isolation_between_gpus():
    m = ThrottleManager({"gpus": [{"gpu_id": 0}, {"gpu_id": 1}]})
    for _ in range(20):
        m.update_window(0, 1.0, 60.0, 0.0)
    assert m.factor_for(0) > 1.0
    assert m.factor_for(1) == 1.0  # unrelated GPU untouched


def test_config_validation():
    validate_throttle_config({"gpus": [{"gpu_id": 0}]})
    with pytest.raises(ValueError):
        validate_throttle_config({"gpus": [{"gpu_id": 0, "tau_s": 0}]})
    with pytest.raises(ValueError):
        validate_throttle_config({"gpus": [{"gpu_id": 0, "th_on": 0.5, "th_off": 0.6}]})
    with pytest.raises(ValueError):
        validate_throttle_config({"gpus": [{"gpu_id": 0, "slowdown_factor": 0.5}]})
    with pytest.raises(ValueError):
        validate_throttle_config({"bogus": 1})
    # alpha > 1 would let L exceed 1.0 ("hotter than fully busy") and move
    # th_on without saying so (F11).
    with pytest.raises(ValueError):
        validate_throttle_config({"gpus": [{"gpu_id": 0, "alpha": 1.5}]})
    with pytest.raises(ValueError):
        validate_throttle_config({"gpus": [{"gpu_id": 0, "alpha": 0.0}]})
    validate_throttle_config({"gpus": [{"gpu_id": 0, "alpha": 0.5}]})


def test_throttle_template_can_fire_inside_a_review_run():
    """A load-coupled fault that never engages is a wasted run, not a result.

    The accumulator is L(t) = u*(1 - exp(-t/tau)). With tau_s=20 and
    th_on=0.55 a run has to stay near u=0.75 for ~28 s before it trips;
    review_min runs are 20-25 s, so the matrix template as first written
    would have produced "neutral" runs and no throttle fault at all. Pin
    the configured tau against the span the matrix actually produces.
    """
    def run(tau, seconds=25.0, window=5.0, u=0.75):
        state = GpuThrottleState(gpu_id=0, tau_s=tau, th_on=0.55,
                                 th_off=0.40, window_s=window)
        edges = []
        for w in range(1, int(seconds / window) + 1):
            edge = state.update(u, window, w * window)
            if edge is not None:
                edges.append(edge)
        return state, edges

    state, edges = run(tau=8.0)
    assert edges and edges[0][1] is True
    assert edges[0][0] <= 20.0

    # The finding this test encodes: tau_s=20 does NOT trip in the same span.
    slow_state, slow_edges = run(tau=20.0)
    assert slow_edges == [] and slow_state.throttled is False
    # underscore-prefixed metadata (e.g. _comment) is allowed
    validate_throttle_config({"_comment": "x", "gpus": [{"gpu_id": 0}]})


def _row(comp, comm_type="ALLREDUCE", comm=1000):
    # 11-field trace row: [layer, comp_time, ..., comm_type, comm_size, ...]
    return ["layer", str(comp), "a", "b", "c", "d", "e", "f",
            comm_type, str(comm), "z"]


def test_apply_factor_scales_compute_only():
    rows = [_row(1000), _row(2000, "NONE", 0),
            ["EXPERT", "marker"], _row(3000)]
    nominal, applied = apply_throttle_factor(rows, 1.5)
    assert nominal == 6000 and applied == 9000
    assert rows[0][1] == "1500" and rows[1][1] == "3000"
    assert rows[0][9] == "1000" and rows[3][9] == "1000"  # comm untouched
    assert rows[2] == ["EXPERT", "marker"]  # non-11-field untouched


def test_apply_factor_neutral_when_unthrottled():
    rows = [_row(1000)]
    nominal, applied = apply_throttle_factor(rows, 1.0)
    assert nominal == applied == 1000


def test_tick_attribution_and_cap():
    mgr = ThrottleManager({"gpus": [{"gpu_id": 0, "tau_s": 600.0,
                                     "window_s": 60.0}]})
    audit = []
    busy = {0: 30_000_000_000}  # 30s of batch time in a 60s window
    tick_throttle(mgr, {0: 0}, busy, audit, 60_000_000_000, 60_000_000_000)
    assert busy == {}  # window drained
    assert len(audit) == 1 and audit[0]["type"] == "throttle_window"
    assert audit[0]["busy_frac"] == pytest.approx(0.5)
    # Overlapping in-flight batches cap at 1.0, never above.
    busy2 = {0: 200_000_000_000}
    tick_throttle(mgr, {0: 0}, busy2, audit, 60_000_000_000, 120_000_000_000)
    assert audit[-1]["busy_frac"] == 1.0


def test_tick_partial_config_ignores_unconfigured_gpus():
    # Regression: ticking with a subset config (e.g. decode pool only)
    # must not KeyError on unconfigured GPUs; their factor stays 1.0.
    mgr = ThrottleManager({"gpus": [{"gpu_id": 8, "window_s": 1.0}]})
    audit = []
    tick_throttle(mgr, {8: 2, 0: 0, 1: 0}, {2: 500_000_000},
                  audit, 1_000_000_000, 1_000_000_000)
    assert len(audit) == 1 and audit[0]["gpu_id"] == 8
    assert audit[0]["instance"] == 2
    assert mgr.factor_for(0) == 1.0


# ---------------------------------------------------------------------------
# benign jitter
# ---------------------------------------------------------------------------
class FakeTrace:
    def __init__(self, rows):
        self.rows = rows


ROWS = [
    ("layer0", "1000000", "LOCAL", "64", "LOCAL", "128", "LOCAL", "64", "NONE", "0", "b0"),
    ("layer1", "2000000", "LOCAL", "64", "LOCAL", "128", "LOCAL", "64", "ALLREDUCE", "4096", "b0"),
    ("layer2", "0", "LOCAL", "64", "LOCAL", "128", "LOCAL", "64", "NONE", "0", "b0"),
]


def test_sigma_zero_is_a_noop():
    trace = FakeTrace(list(ROWS))
    apply_timing_jitter(trace, 0.0, 1234, 0, 1)
    assert trace.rows == ROWS


def test_jitter_is_deterministic_and_keyed():
    a = FakeTrace(list(ROWS))
    b = FakeTrace(list(ROWS))
    apply_timing_jitter(a, 0.10, 1234, 0, 1)
    apply_timing_jitter(b, 0.10, 1234, 0, 1)
    assert a.rows == b.rows
    # A different batch id must produce different noise.
    c = FakeTrace(list(ROWS))
    apply_timing_jitter(c, 0.10, 1234, 0, 2)
    assert c.rows != a.rows


def test_only_latency_and_only_positive_rows_change():
    trace = FakeTrace(list(ROWS))
    apply_timing_jitter(trace, 0.20, 1234, 0, 1)
    for before, after in zip(ROWS, trace.rows):
        # shape / comm / marker fields are untouched
        assert before[0] == after[0]
        assert before[2:] == after[2:]
        if before[1] == "0":
            assert after[1] == "0"   # zero-latency marker rows stay zero
        else:
            assert int(after[1]) > 0


def test_factor_is_mean_preserving():
    values = [jitter_factor(1234, 0, 1, i, 0.25) for i in range(20000)]
    mean = sum(values) / len(values)
    assert abs(mean - 1.0) < 0.02, mean
