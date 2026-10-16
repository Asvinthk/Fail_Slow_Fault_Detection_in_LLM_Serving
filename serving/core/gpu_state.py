"""Per-GPU synthetic accumulated-load throttling (stateful degradation).

SYNTHETIC model, not physical thermal simulation. The state L is a
dimensionless workload-history accumulator, NOT a measured temperature.
A true thermal model would need power, heat dissipation, ambient
conditions, and temperature -- none of which are modelled here.

Update rule (exponential memory), evaluated per fixed sim-time window:
    L += (u - L) * (1 - exp(-dt / tau))     (equivalent to the
    L(t+dt) = L*exp(-dt/tau) + alpha*u*(1-exp(-dt/tau)) form with
    alpha folded into the threshold scale; alpha=1 default so L -> u
    at steady state).

  u ......... per-window GPU busy fraction in [0, 1], computed by the
              caller from ACTUAL simulated batch execution intervals
              (sum of batch durations in the window / window length).
              Never derived from queue length (that would be circular:
              throttling grows queues, queues must not grow L directly).
  tau ....... recovery/accumulation time constant (sim seconds).
  Thresholds with hysteresis: throttle engages at L >= th_on,
              releases at L <= th_off (th_off < th_on).
  Effect: while throttled, compute time for that GPU's future batches
              is scaled by slowdown_factor (>=1), clamped so the
              effective throughput multiplier never drops below m_min.
              Compute rows only -- comm_size is never touched here.

Application timing: schedule-time hook (future batches only). Batches
already handed to ASTRA-Sim are never rescaled -- same documented
limitation as fault_injection.apply_to_trace_rows.

Disabled by default: with no config, update() is never called and
factor() returns 1.0, leaving existing behaviour bit-identical.
"""

import json
import math

DEFAULTS = {
    "tau_s": 600.0,
    "alpha": 1.0,
    "th_on": 0.75,
    "th_off": 0.60,
    "slowdown_factor": 1.5,
    "m_min": 0.5,
    "window_s": 60.0,
}


def validate_throttle_config(config):
    if not isinstance(config, dict):
        raise ValueError("throttle config must be a JSON object")
    unknown = set(config.keys()) - {"gpus", "seed"}
    unknown = {k for k in unknown if not k.startswith("_")}
    if unknown:
        raise ValueError("unknown top-level keys: %s" % sorted(unknown))
    gpus = config.get("gpus", [])
    if not isinstance(gpus, list):
        raise ValueError("'gpus' must be a list")
    seen = set()
    for g in gpus:
        if not isinstance(g, dict):
            raise ValueError("each gpu entry must be an object")
        for key in ("gpu_id",):
            if key not in g:
                raise ValueError("gpu entry missing keys: %s" % key)
        allowed = {"gpu_id", "tau_s", "alpha", "th_on", "th_off",
                   "slowdown_factor", "m_min", "window_s"}
        bad = set(g.keys()) - allowed
        if bad:
            raise ValueError("gpu %s has unknown keys: %s" % (g.get("gpu_id"), sorted(bad)))
        if g["gpu_id"] in seen:
            raise ValueError("duplicate gpu_id: %s" % g["gpu_id"])
        seen.add(g["gpu_id"])
        tau = g.get("tau_s", DEFAULTS["tau_s"])
        th_on = g.get("th_on", DEFAULTS["th_on"])
        th_off = g.get("th_off", DEFAULTS["th_off"])
        slow = g.get("slowdown_factor", DEFAULTS["slowdown_factor"])
        mmin = g.get("m_min", DEFAULTS["m_min"])
        if tau <= 0:
            raise ValueError("tau_s must be > 0 for %s" % g["gpu_id"])
        if not (0 < th_off < th_on <= 1.0 + 1e-9):
            raise ValueError("need 0 < th_off < th_on <= ~1 for %s" % g["gpu_id"])
        if slow < 1.0:
            raise ValueError("slowdown_factor must be >= 1 for %s" % g["gpu_id"])
        if not (0 < mmin <= 1.0):
            raise ValueError("m_min must be in (0, 1] for %s" % g["gpu_id"])
        if 1.0 / slow < mmin - 1e-9:
            raise ValueError("slowdown_factor violates m_min for %s" % g["gpu_id"])


def load_throttle_config(path):
    with open(path, "r") as f:
        config = json.load(f)
    validate_throttle_config(config)
    return config


class GpuThrottleState:
    """Independent accumulated-load + hysteresis state for one GPU."""

    def __init__(self, gpu_id, tau_s=600.0, alpha=1.0, th_on=0.75,
                 th_off=0.60, slowdown_factor=1.5, m_min=0.5,
                 window_s=60.0):
        self.gpu_id = gpu_id
        self.tau_s = tau_s
        self.alpha = alpha
        self.th_on = th_on
        self.th_off = th_off
        self.slowdown_factor = slowdown_factor
        self.m_min = m_min
        self.window_s = window_s
        self.load = 0.0
        self.throttled = False
        self.transitions = []  # (window_end_s, throttled, load) ground truth

    def update(self, busy_frac, dt_s, window_end_s):
        """Advance L with one window observation; returns throttle edge or None."""
        u = max(0.0, min(1.0, busy_frac)) * self.alpha
        w = 1.0 - math.exp(-dt_s / self.tau_s)
        self.load += (u - self.load) * w
        prev = self.throttled
        if not self.throttled and self.load >= self.th_on:
            self.throttled = True
        elif self.throttled and self.load <= self.th_off:
            self.throttled = False
        if self.throttled != prev:
            self.transitions.append((window_end_s, self.throttled, self.load))
            return (window_end_s, self.throttled)
        return None

    def factor(self):
        """Compute-time multiplier for this GPU's future batches (>=1)."""
        if not self.throttled:
            return 1.0
        return min(self.slowdown_factor, 1.0 / self.m_min)

    def audit_row(self, window_end_s, busy_frac):
        return {"gpu_id": self.gpu_id, "window_end_s": window_end_s,
                "busy_frac": busy_frac, "load": self.load,
                "throttled": self.throttled, "factor": self.factor()}


def apply_throttle_factor(rows, factor):
    """Scale comp_time of 11-field trace rows by factor (>=1).

    Compute rows only -- comm_size is never touched (compute throttling
    must not masquerade as network degradation). Returns
    (nominal_comp_sum, applied_comp_sum). Integer rounding matches the
    Chakra converter requirement, same as fault_injection.
    """
    COL_COMP_TIME = 1
    nominal = 0
    applied = 0
    for idx, old_row in enumerate(rows):
        if len(old_row) != 11:
            continue
        rows[idx] = list(old_row)
    for row in rows:
        if len(row) != 11:
            continue
        old = int(row[COL_COMP_TIME])
        nominal += old
        new = int(round(old * factor))
        row[COL_COMP_TIME] = str(new)
        applied += new
    return nominal, applied


class ThrottleManager:
    """Owns one GpuThrottleState per configured GPU id."""

    def __init__(self, config=None):
        self.states = {}
        self.window_s = DEFAULTS["window_s"]
        if config:
            for g in config.get("gpus", []):
                gid = g["gpu_id"]
                kw = {k: g.get(k, DEFAULTS[k]) for k in DEFAULTS}
                self.states[gid] = GpuThrottleState(gpu_id=gid, **kw)
                self.window_s = kw["window_s"]

    def enabled(self):
        return bool(self.states)

    def factor_for(self, gpu_id):
        st = self.states.get(gpu_id)
        return st.factor() if st is not None else 1.0

    def update_window(self, gpu_id, busy_frac, dt_s, window_end_s):
        st = self.states.get(gpu_id)
        if st is None:
            return None
        return st.update(busy_frac, dt_s, window_end_s)
