"""Seeded multiplicative jitter on per-layer compute time (benign noise).

Why this exists (review item N2): the simulator's timing is a
deterministic table lookup, so healthy residuals have almost zero
variance. Any injected fault -- even 1.1x -- is then trivially
detectable, and M2 (false alarms under benign variation) cannot be
measured at all. Real serving has run-to-run variance from clock drift,
DVFS, scheduler jitter and profiler noise; this module injects it.

Properties that matter for the experiment:

  * OFF BY DEFAULT. sigma == 0 returns the trace untouched, so
    ``Total clocks (ns)`` stays bit-identical to the no-jitter baseline
    and the 62-scenario regression suite keeps passing.
  * SEEDED AND KEYED, not streamed. The multiplier for a row is derived
    from (seed, instance_id, batch_id, row_index) through a hash, so it
    does not depend on evaluation order, on how many rows came before,
    or on whether a fault is active. Rerunning the same config gives
    the same noise.
  * MEAN-PRESERVING. The multiplier is lognormal with median 1 and
    E[factor] = exp(-sigma^2/2) * exp(sigma^2/2) = 1, so enabling
    jitter does not systematically inflate or deflate latency -- only
    its spread.
  * COMPUTE ONLY. Only the latency field of a trace row is touched;
    communication sizes, weight bytes and shapes are untouched, so a
    jittered run cannot masquerade as a transfer fault.

sigma is a sweep parameter (0.02, 0.05, 0.10, ...), not a calibrated
physical constant: the profiler bundles do not record repeat-run
variance, so there is nothing to calibrate against yet. Detectability is
reported as a function of sigma rather than at one invented value.
"""

import hashlib
import math

# Trace row layout (trace_generator.generate_trace): index 1 is the
# layer latency in ns, as a string.
LATENCY_INDEX = 1


def _uniform(seed, instance_id, batch_id, row_idx, salt):
    """Deterministic U(0,1) from the row key (open interval, never 0)."""
    key = ("%d|%d|%d|%d|%d" % (seed, instance_id, batch_id, row_idx, salt)).encode()
    digest = hashlib.blake2b(key, digest_size=8).digest()
    return (int.from_bytes(digest, "big") + 0.5) / float(1 << 64)


def factor_for(seed, instance_id, batch_id, row_idx, sigma):
    """Lognormal multiplier with mean 1 and log-sd sigma."""
    if sigma <= 0:
        return 1.0
    u1 = _uniform(seed, instance_id, batch_id, row_idx, 1)
    u2 = _uniform(seed, instance_id, batch_id, row_idx, 2)
    z = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
    return math.exp(sigma * z - 0.5 * sigma * sigma)


def apply_timing_jitter(trace_data, sigma, seed, instance_id, batch_id):
    """Scale each compute row's latency by a seeded lognormal factor.

    Rows are tuples (fault_injection copies them before scaling, and the
    graph writer accepts tuples), so a jittered row is rebuilt rather
    than mutated in place.
    """
    if sigma <= 0:
        return trace_data
    rows = getattr(trace_data, "rows", None)
    if rows is None:
        return trace_data
    jittered = []
    for idx, row in enumerate(rows):
        if isinstance(row, (tuple, list)) and len(row) > LATENCY_INDEX:
            try:
                latency = float(row[LATENCY_INDEX])
            except (TypeError, ValueError):
                latency = None
            if latency is not None and latency > 0:
                factor = factor_for(seed, instance_id, batch_id, idx, sigma)
                row = tuple(row)
                row = (row[:LATENCY_INDEX]
                       + (str(max(1, int(round(latency * factor)))),)
                       + row[LATENCY_INDEX + 1:])
        jittered.append(row)
    trace_data.rows = jittered
    return trace_data
