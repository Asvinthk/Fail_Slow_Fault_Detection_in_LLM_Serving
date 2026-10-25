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


def test_thinning_bound_covers_peak_plus_burst():
    """F4: the bound must be the supremum of the FULL intensity.

    The old bound was max(peak, burst) taken separately, which is not an
    upper bound when a burst overlaps the noon peak: rate/max > 1, every
    candidate is accepted and the process saturates at the bound.
    """
    from experiments.fault_detection.day_workload import intensity_at, max_intensity
    cfg = _cfg(horizon_h=1.0, peak_hour=0.5, peak_sigma_h=0.1)
    # Burst straddling the peak, exactly the case the old bound missed.
    cfg["bursts"] = [{"start_s": 0.5 * 3600 - 300, "dur_s": 600.0,
                      "rate_mult": 2.0}]
    grid_max = max(intensity_at(t, cfg, cfg["bursts"])
                   for t in range(0, 3600))
    old_bound = max(cfg["base_sps"] * (1.0 + cfg["peak_height"]),
                    cfg["base_sps"] * 2.0)
    assert grid_max > old_bound            # the old bound was invalid here
    assert max_intensity(cfg, cfg["bursts"]) >= grid_max - 1e-9


def test_realised_burst_volume_matches_analytic():
    """A burst must add ~rate_mult*base*dur requests, not a handful.

    Regression for the measured failure: a 600 s x2 burst at the peak
    added 22 requests where the model implies ~1200.
    """
    from experiments.fault_detection.day_workload import intensity_at
    cfg = _cfg(horizon_h=3.0, peak_hour=1.5, peak_sigma_h=0.3)
    start, dur = 1.5 * 3600 - 900, 1800.0
    cfg["bursts"] = [{"start_s": start, "dur_s": dur, "rate_mult": 2.0}]
    rows, _info = generate(cfg)
    inside = sum(1 for r in rows
                 if start <= r["arrival_time_ns"] / 1e9 < start + dur)
    expected = sum(intensity_at(start + i, cfg, cfg["bursts"])
                   for i in range(int(dur)))
    assert abs(inside - expected) / expected < 0.05, (inside, expected)


def test_day_boundary_is_documented_not_periodic():
    """F5: the profile is a ONE-SHOT day; pin the discontinuity.

    If this ever becomes periodic, the docstring and report must change
    with it -- the test exists so the property cannot drift silently.
    """
    cfg = _cfg(horizon_h=24.0, peak_hour=12.0, peak_sigma_h=1.5)
    assert rate_at(23.99 * 3600, cfg) < 0.1 * rate_at(0.1 * 3600, cfg)
