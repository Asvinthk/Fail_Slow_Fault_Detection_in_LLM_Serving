"""Host-only tests for synthetic GPU throttle state (no simulator binary)."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

from serving.core.gpu_state import (
    GpuThrottleState, ThrottleManager, apply_throttle_factor,
    validate_throttle_config)


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
    import serving.__main__ as m
    mgr = ThrottleManager({"gpus": [{"gpu_id": 0, "tau_s": 600.0,
                                     "window_s": 60.0}]})
    audit = []
    busy = {0: 30_000_000_000}  # 30s of batch time in a 60s window
    m._throttle_tick(mgr, {0: 0}, busy, audit, 60_000_000_000,
                     60_000_000_000)
    assert busy == {}  # window drained
    assert len(audit) == 1 and audit[0]["type"] == "throttle_window"
    assert audit[0]["busy_frac"] == pytest.approx(0.5)
    # Overlapping in-flight batches cap at 1.0, never above.
    busy2 = {0: 200_000_000_000}
    m._throttle_tick(mgr, {0: 0}, busy2, audit, 60_000_000_000,
                     120_000_000_000)
    assert audit[-1]["busy_frac"] == 1.0


def test_tick_partial_config_ignores_unconfigured_gpus():
    # Regression: ticking with a subset config (e.g. decode pool only)
    # must not KeyError on unconfigured GPUs; their factor stays 1.0.
    import serving.__main__ as m
    mgr = ThrottleManager({"gpus": [{"gpu_id": 8, "window_s": 1.0}]})
    audit = []
    m._throttle_tick(mgr, {8: 2, 0: 0, 1: 0}, {2: 500_000_000},
                     audit, 1_000_000_000, 1_000_000_000)
    assert len(audit) == 1 and audit[0]["gpu_id"] == 8
    assert audit[0]["instance"] == 2
    assert mgr.factor_for(0) == 1.0
