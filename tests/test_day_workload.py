"""Host-only tests for the day workload generator (no simulator binary)."""
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from experiments.fault_detection.day_workload import default_config, generate, rate_at


def _cfg(**kw):
    args = type("A", (), {})()
    for k, v in {"horizon_h": 2.0, "base_sps": 2.0, "peak_hour": 1.0,
                 "peak_height": 3.0, "peak_sigma_h": 0.25,
                 "afternoon_tau_h": 3.0, "evening_frac": 0.4,
                 "bursts": 0, "burst_dur_s": 60.0, "burst_rate_mult": 2.0,
                 "seed": 7, "short_in": 32, "long_in": 128,
                 "short_out": 16, "long_out": 64}.items():
        setattr(args, k, v)
    for k, v in kw.items():
        setattr(args, k, v)
    return default_config(args)


def test_schema_and_order():
    rows, _ = generate(_cfg())
    assert len(rows) > 0
    prev = -1
    for r in rows:
        assert set(r.keys()) == {"input_toks", "output_toks", "arrival_time_ns"}
        assert r["arrival_time_ns"] >= prev
        prev = r["arrival_time_ns"]
        assert 32 <= r["input_toks"] <= 128


def test_deterministic():
    assert generate(_cfg())[0] == generate(_cfg())[0]
    assert generate(_cfg(seed=8))[0] != generate(_cfg())[0]


def test_peak_shape():
    cfg = _cfg()
    r_peak = rate_at(1.0 * 3600, cfg)
    r_early = rate_at(600, cfg)
    assert r_peak > 2.0 * r_early  # peak_height=3 must dominate
    cfg_off = _cfg(peak_height=0.0)
    assert rate_at(1.0 * 3600, cfg_off) < r_peak  # peak disabled


def test_bursts_add_load():
    a, _ = generate(_cfg(bursts=0))
    b, info = generate(_cfg(bursts=3))
    assert len(info["bursts"]) == 3
    assert len(b) > len(a)
