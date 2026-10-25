"""Tests for the seeded benign-noise jitter (review N2).

The contract that matters for the regression suite: sigma == 0 must be a
byte-for-byte no-op, so `Total clocks (ns)` cannot move.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from serving.core.timing_jitter import (  # noqa: E402
    apply_timing_jitter,
    factor_for,
)


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
    values = [factor_for(1234, 0, 1, i, 0.25) for i in range(20000)]
    mean = sum(values) / len(values)
    assert abs(mean - 1.0) < 0.02, mean
