"""Fault mechanisms, synthetic throttling, validation and audit generation.

This is the single home for every *injected* deviation in the simulator.
It replaces three earlier modules (`fault_injection.py`, `gpu_state.py`,
`timing_jitter.py`) whose boundaries were historical rather than logical:
all three answer the same question -- "how does this batch's trace differ
from nominal, and what is the ground truth for that difference?" -- and
all three must keep the same invariants (future batches only, compute vs
comm never confused, ground truth never merged into telemetry).

Three mechanism families live here:

  injection  (fail-slow)  scale comp_time or comm_size for the targeted
                          instance while a schedule says so. Modes:
                          compute_multiplicative, compute_additive,
                          kv_transfer_bandwidth, kv_handoff_delay,
                          network_degradation, worker_stall.
  throttle   (thermal-   per-GPU accumulated-load state with hysteresis;
              like)       while engaged, compute time of that GPU's future
                          batches is scaled. Load-coupled by design: onset
                          depends on workload, which makes it the useful
                          test of workload-vs-fault discrimination.
  jitter     (benign)    seeded, mean-preserving lognormal multiplier on
                          compute latency. OFF by default; it exists so
                          false-positive rates can be measured against
                          something other than a deterministic table lookup.

Invariants that the tests pin:

  * All three act at SCHEDULE time on FUTURE batches only. A batch already
    handed to ASTRA-Sim is immutable, so onset can be late by up to one
    iteration. Documented, not hidden.
  * Compute faults never touch comm_size and transfer faults never touch
    comp_time, so a fault cannot masquerade as another family.
  * sigma == 0 (jitter) is a byte-for-byte no-op: `Total clocks (ns)`
    cannot move, which keeps the 62-scenario regression suite valid.
  * Every applied change emits an audit record carrying nominal and
    applied sums. `applied=False` is a first-class outcome ("mechanism
    neutral for this batch"), never silently reported as applied.
  * Audit files are ground truth and are written next to telemetry, never
    into it (see experiments/fault_detection/evaluate.py for the leak
    guard).

These are SIMULATOR modes. They say nothing about real hardware causes
(thermal, ECC, PCIe); synthetic faults validate the detection machinery,
never real-hardware detection.
"""

import hashlib
import json
import math

# ---------------------------------------------------------------------------
# trace row layout
# ---------------------------------------------------------------------------
# One layer row has 11 fields (serving/core/utils.py, trace_generator.formatter).
COL_LAYER = 0
COL_COMP_TIME = 1
COL_COMM_TYPE = 8
COL_COMM_SIZE = 9
ROW_FIELDS = 11

# ---------------------------------------------------------------------------
# injection: schema and validation
# ---------------------------------------------------------------------------
ALLOWED_MODES = (
    "compute_multiplicative",
    "compute_additive",
    "kv_transfer_bandwidth",
    "kv_handoff_delay",
    "network_degradation",
    "worker_stall",
)

ALLOWED_ROLES = ("prefill", "decode", "transfer", "network")

ALLOWED_PROFILES = ("step", "intermittent", "ramp_up", "ramp_down", "ramp_up_down")

FAILURE_CLASS = {
    "compute_multiplicative": "fail_slow",
    "compute_additive": "fail_slow",
    "kv_transfer_bandwidth": "fail_slow",
    "kv_handoff_delay": "fail_slow",
    "network_degradation": "fail_slow",
    "worker_stall": "fail_stop_stall",
}


def load_fault_config(path):
    """Read and validate a fault JSON file. Returns a plain dict."""
    with open(path, "r") as f:
        config = json.load(f)
    validate_fault_config(config)
    return config


def validate_fault_config(config):
    """Reject unknown keys and bad combos at startup with clear errors."""
    if not isinstance(config, dict):
        raise ValueError("fault config must be a JSON object")
    top_keys = set(config.keys())
    unknown = top_keys - {"faults", "seed"}
    if unknown:
        raise ValueError("unknown top-level keys: %s" % sorted(unknown))
    faults = config.get("faults", [])
    if not isinstance(faults, list):
        raise ValueError("'faults' must be a list")
    seen_ids = set()
    for fault in faults:
        validate_one_fault(fault, seen_ids)
        seen_ids.add(fault["fault_id"])


def validate_one_fault(fault, seen_ids):
    if not isinstance(fault, dict):
        raise ValueError("each fault must be an object")
    required = {"fault_id", "target", "injection_layer", "mode", "schedule"}
    missing = required - set(fault.keys())
    if missing:
        raise ValueError("fault missing keys: %s" % sorted(missing))
    allowed = required | {"factor", "delta_ns", "bandwidth_fraction",
                          "delay_ns", "link_bw_scale", "failure_class"}
    unknown = set(fault.keys()) - allowed
    if unknown:
        raise ValueError("fault %s has unknown keys: %s"
                         % (fault.get("fault_id"), sorted(unknown)))
    if fault["fault_id"] in seen_ids:
        raise ValueError("duplicate fault_id: %s" % fault["fault_id"])
    target = fault["target"]
    if set(target.keys()) != {"role", "instance_id"}:
        raise ValueError("target must have exactly role and instance_id")
    if target["role"] not in ALLOWED_ROLES:
        raise ValueError("bad role: %s" % target["role"])
    if fault["mode"] not in ALLOWED_MODES:
        raise ValueError("bad mode: %s" % fault["mode"])
    if fault["mode"] == "compute_multiplicative" and "factor" not in fault:
        raise ValueError("compute_multiplicative needs 'factor'")
    if fault["mode"] == "compute_additive" and "delta_ns" not in fault:
        raise ValueError("compute_additive needs 'delta_ns'")
    if fault["mode"] == "kv_transfer_bandwidth" and "bandwidth_fraction" not in fault:
        raise ValueError("kv_transfer_bandwidth needs 'bandwidth_fraction'")
    if fault["mode"] == "kv_handoff_delay" and "delay_ns" not in fault:
        raise ValueError("kv_handoff_delay needs 'delay_ns'")
    if fault["mode"] == "worker_stall":
        if "factor" in fault or "delta_ns" in fault:
            raise ValueError("worker_stall takes no factor or delta_ns")
    schedule = fault["schedule"]
    if set(schedule.keys()) - {"start_s", "end_s", "profile", "period_s",
                               "duty", "ramp_s"}:
        raise ValueError("schedule has unknown keys for %s" % fault["fault_id"])
    if "start_s" not in schedule or "end_s" not in schedule:
        raise ValueError("schedule needs start_s and end_s")
    if schedule["end_s"] <= schedule["start_s"]:
        raise ValueError("schedule end_s must be after start_s")
    profile = schedule.get("profile", "step")
    if profile not in ALLOWED_PROFILES:
        raise ValueError("bad profile: %s" % profile)
    if fault.get("injection_layer") == "output_approx" \
            and fault["mode"] != "network_degradation":
        raise ValueError("output_approx is only allowed for network_degradation (latency)")


# ---------------------------------------------------------------------------
# injection: schedule and targeting
# ---------------------------------------------------------------------------
def severity_weight(schedule, now_s):
    """Severity w(t) in [0, 1] for sim time now_s (seconds)."""
    start = schedule["start_s"]
    end = schedule["end_s"]
    profile = schedule.get("profile", "step")
    if now_s < start or now_s >= end:
        return 0.0
    if profile == "step":
        return 1.0
    if profile == "intermittent":
        period = schedule.get("period_s", 1.0)
        duty = schedule.get("duty", 0.5)
        phase = (now_s - start) % period
        return 1.0 if phase < duty * period else 0.0
    if profile == "ramp_up":
        ramp = schedule.get("ramp_s", end - start)
        if ramp <= 0:
            return 1.0
        return min(1.0, (now_s - start) / ramp)
    if profile == "ramp_down":
        ramp = schedule.get("ramp_s", end - start)
        if ramp <= 0:
            return 1.0
        return min(1.0, (end - now_s) / ramp)
    if profile == "ramp_up_down":
        ramp = schedule.get("ramp_s", (end - start) / 2.0)
        if ramp <= 0:
            return 1.0
        return max(0.0, min(1.0, (now_s - start) / ramp, (end - now_s) / ramp))
    return 1.0


def fault_matches(fault, role, instance_id):
    """True if this fault targets the given batch role and instance."""
    target = fault["target"]
    target_role = target["role"]
    target_inst = target["instance_id"]
    if target_role == "network":
        # Network faults target an instance regardless of its role.
        if target_inst == "all":
            return True
        return target_inst == instance_id
    role_match = (target_role == role) or (target_role == "transfer" and role == "prefill")
    if not role_match:
        return False
    if target_inst == "all":
        return True
    return target_inst == instance_id


def is_stalled(faults, role, instance_id, now_s):
    """True if a worker_stall fault covers this instance right now."""
    for fault in faults:
        if fault["mode"] != "worker_stall":
            continue
        target = fault["target"]
        if target["instance_id"] != "all" and target["instance_id"] != instance_id:
            continue
        if target["role"] != role:
            continue
        if severity_weight(fault["schedule"], now_s) > 0:
            return True
    return False


# ---------------------------------------------------------------------------
# injection: trace scaling
# ---------------------------------------------------------------------------
def apply_to_trace_rows(faults, rows, role, instance_id, now_s):
    """Scale comp_time and comm_size of trace rows. Returns audit entries.

    Called after generate_trace built the rows and before graph conversion,
    so the graph-cache digest changes naturally. Integer rounding is applied
    because the Chakra converter requires int comp_time.

    Semantics per mode:
      compute_multiplicative: every compute row scaled by the effective
        factor 1 + w(t)*(factor-1).
      compute_additive: per-ITERATION delay. delta_ns (weighted by w(t))
        is added to the comp_time of ONE designated serial row (the last
        11-field compute row of the chain), not to every layer. Adding it
        per layer would multiply the delay by the layer count.
      kv_transfer_bandwidth: P/D comm_size on qkv_proj rows inflated by
        the effective factor 1 + w(t)*(1/fraction - 1). Models only the
        bandwidth term, not latency.
      network_degradation (mechanism): time-varying scaling of comm_size
        on every row carrying a collective (comm_type != NONE with
        comm_size != 0), for the targeted instance. Time-varying link
        LATENCY is not modeled here (needs ASTRA-Sim changes).

    worker_stall and kv_handoff_delay are handled by the event loop, not by
    row scaling. Rows that are not 11-field layer rows (MoE EXPERT/PIM
    markers) are left untouched.
    """
    audits = []
    for fault in faults:
        if fault["mode"] in ("worker_stall", "kv_handoff_delay"):
            continue
        if not fault_matches(fault, role, instance_id):
            continue
        weight = severity_weight(fault["schedule"], now_s)
        if weight <= 0:
            continue
        mode = fault["mode"]
        nominal_comp = 0
        applied_comp = 0
        nominal_comm = 0
        applied_comm = 0
        # First pass: copy tuples to lists so scaling never crashes.
        for idx, old_row in enumerate(rows):
            if len(old_row) != ROW_FIELDS:
                continue
            rows[idx] = list(old_row)
        if mode == "compute_additive":
            nominal_comp = sum(int(r[COL_COMP_TIME]) for r in rows
                               if len(r) == ROW_FIELDS)
            for idx in range(len(rows) - 1, -1, -1):
                if len(rows[idx]) != ROW_FIELDS:
                    continue
                old = int(rows[idx][COL_COMP_TIME])
                new = int(old + round(weight * fault["delta_ns"]))
                rows[idx][COL_COMP_TIME] = str(new)
                break
            applied_comp = (nominal_comp
                            + int(round(weight * fault["delta_ns"])))
        else:
            for row in rows:
                if len(row) != ROW_FIELDS:
                    continue
                layer = row[COL_LAYER]
                if mode == "compute_multiplicative":
                    old = int(row[COL_COMP_TIME])
                    nominal_comp += old
                    factor = 1.0 + weight * (fault["factor"] - 1.0)
                    new = int(round(old * factor))
                    row[COL_COMP_TIME] = str(new)
                    applied_comp += new
                elif mode == "network_degradation":
                    if fault.get("injection_layer") == "output_approx":
                        continue
                    old = int(row[COL_COMP_TIME])
                    nominal_comp += old
                    applied_comp += old
                    comm_type = str(row[COL_COMM_TYPE])
                    try:
                        old_comm = int(row[COL_COMM_SIZE])
                    except (ValueError, TypeError):
                        old_comm = 0
                    if comm_type != "NONE" and old_comm != 0:
                        nominal_comm += old_comm
                        scale = 1.0 + weight * (fault.get("factor", 1.5) - 1.0)
                        new_comm = int(round(old_comm * scale))
                        row[COL_COMM_SIZE] = str(new_comm)
                        applied_comm += new_comm
                if mode == "kv_transfer_bandwidth":
                    # Only qkv_proj rows carry the P/D KV transfer in comm_size.
                    if layer.startswith("qkv_proj") and role == "prefill":
                        old_comm = int(row[COL_COMM_SIZE])
                        nominal_comm += old_comm
                        frac = fault["bandwidth_fraction"]
                        eff = 1.0 + weight * (1.0 / frac - 1.0)
                        new_comm = int(round(old_comm * eff))
                        row[COL_COMM_SIZE] = str(new_comm)
                        applied_comm += new_comm
        changed = (applied_comp != nominal_comp
                   or applied_comm != nominal_comm)
        entry = {
            "fault_id": fault["fault_id"],
            "mode": mode,
            "weight": weight,
            "nominal_comp_sum": nominal_comp,
            "applied_comp_sum": applied_comp,
            "nominal_comm_sum": nominal_comm,
            "applied_comm_sum": applied_comm,
            "applied": changed,
        }
        if not changed:
            # Active window, visited rows, but nothing to scale (e.g.
            # network_degradation on a tp=1 trace has no collective rows).
            # Never silently label it applied.
            entry["note"] = ("no matching rows changed; mechanism neutral "
                             "for this batch")
        audits.append(entry)
    return audits


# ---------------------------------------------------------------------------
# injection: handoff hold (transfer-side, event keyed)
# ---------------------------------------------------------------------------
def select_handoff_hold(faults, role, instance_id, now_s):
    """Strongest active kv_handoff_delay for this instance.

    Returns (delay_ns, fault) or (0, None). The hold is a transfer-link
    event owned by the PREFILL instance that ships the KV, which is why
    the audit and the labels attribute it to that instance.
    """
    delay = 0
    chosen = None
    for fault in faults:
        if fault["mode"] != "kv_handoff_delay":
            continue
        if not fault_matches(fault, role, instance_id):
            continue
        candidate = int(round(severity_weight(fault["schedule"], now_s)
                              * fault["delay_ns"]))
        if candidate > delay:
            delay = candidate
            chosen = fault
    return delay, chosen


def audit_handoff_hold(fault, instance_id, time_ns, delay_ns, count, weight):
    """Audit row for a held KV handoff (event keyed: no batch id)."""
    return {
        "type": "handoff_hold",
        "fault_id": fault["fault_id"],
        "mode": "kv_handoff_delay",
        "weight": weight,
        "instance": instance_id,
        "time_ns": time_ns,
        "delay_ns": delay_ns,
        "count": count,
        "nominal_comp_sum": 0, "applied_comp_sum": 0,
        "nominal_comm_sum": 0, "applied_comm_sum": 0,
        "applied": True,
    }


# ---------------------------------------------------------------------------
# injection: worker stall (absence of batches is the evidence)
# ---------------------------------------------------------------------------
def note_stall_skip(stall_skips, faults, role, instance_id, now_s):
    """Count one skipped schedule poll per active stall fault."""
    for fault in faults:
        if fault["mode"] != "worker_stall":
            continue
        if not fault_matches(fault, role, instance_id):
            continue
        if severity_weight(fault["schedule"], now_s) > 0:
            key = (fault["fault_id"], instance_id)
            stall_skips[key] = stall_skips.get(key, 0) + 1


def audit_stall_windows(stall_skips, faults):
    """One summary row per (fault, instance): skipped polls, not batches."""
    rows = []
    for (fault_id, instance_id), skipped in stall_skips.items():
        fault = next(f for f in faults if f["fault_id"] == fault_id)
        rows.append({
            "type": "worker_stall_window",
            "fault_id": fault_id,
            "mode": "worker_stall",
            "weight": 1.0,
            "instance": instance_id,
            "time_ns": int(fault["schedule"]["start_s"] * 1e9),
            "skipped_schedules": skipped,
            "nominal_comp_sum": 0, "applied_comp_sum": 0,
            "nominal_comm_sum": 0, "applied_comm_sum": 0,
            "applied": skipped > 0,
        })
    return rows


def attach_batch_context(entry, batch_id, instance_id, time_ns):
    """Stamp a trace-scaling audit entry with the batch it belongs to."""
    entry.update({"batch_id": batch_id, "instance": instance_id,
                  "time_ns": time_ns})
    return entry


# ---------------------------------------------------------------------------
# throttle: schema, state, application
# ---------------------------------------------------------------------------
THROTTLE_DEFAULTS = {
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
        if "gpu_id" not in g:
            raise ValueError("gpu entry missing keys: gpu_id")
        allowed = {"gpu_id", "tau_s", "alpha", "th_on", "th_off",
                   "slowdown_factor", "m_min", "window_s"}
        bad = set(g.keys()) - allowed
        if bad:
            raise ValueError("gpu %s has unknown keys: %s"
                             % (g.get("gpu_id"), sorted(bad)))
        if g["gpu_id"] in seen:
            raise ValueError("duplicate gpu_id: %s" % g["gpu_id"])
        seen.add(g["gpu_id"])
        tau = g.get("tau_s", THROTTLE_DEFAULTS["tau_s"])
        alpha = g.get("alpha", THROTTLE_DEFAULTS["alpha"])
        th_on = g.get("th_on", THROTTLE_DEFAULTS["th_on"])
        th_off = g.get("th_off", THROTTLE_DEFAULTS["th_off"])
        slow = g.get("slowdown_factor", THROTTLE_DEFAULTS["slowdown_factor"])
        mmin = g.get("m_min", THROTTLE_DEFAULTS["m_min"])
        if tau <= 0:
            raise ValueError("tau_s must be > 0 for %s" % g["gpu_id"])
        # alpha scales the observed busy fraction before it enters L. Above 1
        # it can push L past 1.0, i.e. "hotter than fully busy", which has no
        # meaning in this synthetic model and silently moves th_on.
        if not (0 < alpha <= 1.0):
            raise ValueError("alpha must be in (0, 1] for %s" % g["gpu_id"])
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
    """Independent accumulated-load + hysteresis state for one GPU.

    SYNTHETIC model, not physical thermal simulation. The state L is a
    dimensionless workload-history accumulator, NOT a measured
    temperature. Update rule (exponential memory), per fixed sim window:

        L += (u - L) * (1 - exp(-dt / tau))

    u is the GPU busy fraction computed by the caller from ACTUAL batch
    execution intervals -- never from queue length (throttling grows
    queues, so queues must not grow L; that would be circular).
    """

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
    (nominal_comp_sum, applied_comp_sum).
    """
    nominal = 0
    applied = 0
    for idx, old_row in enumerate(rows):
        if len(old_row) != ROW_FIELDS:
            continue
        rows[idx] = list(old_row)
    for row in rows:
        if len(row) != ROW_FIELDS:
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
        self.window_s = THROTTLE_DEFAULTS["window_s"]
        if config:
            for g in config.get("gpus", []):
                gid = g["gpu_id"]
                kw = {k: g.get(k, THROTTLE_DEFAULTS[k])
                      for k in THROTTLE_DEFAULTS}
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


def tick_throttle(throttle_manager, gpu_to_instance, window_busy, audit,
                  window_ns, window_end_ns):
    """Advance every configured GPU by one fixed sim-time window.

    Busy fractions come from ACTUAL completed-batch durations attributed to
    the window in which the batch completes (capped at 1.0 for overlapping
    in-flight batches). Under lockstep SPMD every member NPU of an instance
    shares the batch duration, so per-instance busy time applies to each of
    its physical GPUs. Appends one ground-truth row per configured GPU.
    """
    window_s = window_ns / 1e9
    window_end_s = window_end_ns / 1e9
    for gpu_id in throttle_manager.states:
        instance_id = gpu_to_instance.get(gpu_id)
        busy_ns = window_busy.get(instance_id, 0) if instance_id is not None else 0
        busy_frac = min(1.0, busy_ns / window_ns) if window_ns > 0 else 0.0
        edge = throttle_manager.update_window(gpu_id, busy_frac,
                                              window_s, window_end_s)
        row = throttle_manager.states[gpu_id].audit_row(window_end_s, busy_frac)
        row["type"] = "throttle_window"
        row["instance"] = instance_id
        # A window is "applied" when this GPU was throttled for it. Without
        # this flag a window row is indistinguishable from a neutral one,
        # and the audit consistency check would report a silent absence.
        row["applied"] = row["factor"] > 1.0
        if edge is not None:
            row["edge"] = edge[1]
        audit.append(row)
    window_busy.clear()


def audit_throttle_batch(instance_id, batch_id, time_ns, factor,
                         nominal, applied):
    """Ground-truth row for one throttled batch (compute-only scaling)."""
    return {
        "type": "throttle_batch",
        "instance": instance_id,
        "batch_id": batch_id,
        "time_ns": time_ns,
        "factor": factor,
        "nominal_comp_sum": nominal,
        "applied_comp_sum": applied,
        "applied": True,
    }


# ---------------------------------------------------------------------------
# benign noise: seeded mean-preserving lognormal jitter on compute time
# ---------------------------------------------------------------------------
def _uniform(seed, instance_id, batch_id, row_idx, salt):
    """Deterministic U(0,1) from the row key (open interval, never 0)."""
    key = ("%d|%d|%d|%d|%d" % (seed, instance_id, batch_id, row_idx,
                               salt)).encode()
    digest = hashlib.blake2b(key, digest_size=8).digest()
    return (int.from_bytes(digest, "big") + 0.5) / float(1 << 64)


def jitter_factor(seed, instance_id, batch_id, row_idx, sigma):
    """Lognormal multiplier with E[factor] = 1 and log-sd sigma.

    Keyed rather than streamed: the multiplier depends only on
    (seed, instance, batch, row), so it is order-independent, identical on
    rerun, and identical whether or not a fault is active.
    """
    if sigma <= 0:
        return 1.0
    u1 = _uniform(seed, instance_id, batch_id, row_idx, 1)
    u2 = _uniform(seed, instance_id, batch_id, row_idx, 2)
    z = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
    return math.exp(sigma * z - 0.5 * sigma * sigma)


def apply_timing_jitter(trace_data, sigma, seed, instance_id, batch_id):
    """Scale each compute row's latency by a seeded lognormal factor.

    sigma == 0 returns the trace untouched, so `Total clocks (ns)` stays
    bit-identical to the no-jitter baseline. Only the latency field is
    touched: shapes, comm sizes and markers are never jittered, so noise
    cannot masquerade as a transfer fault.
    """
    if sigma <= 0:
        return trace_data
    rows = getattr(trace_data, "rows", None)
    if rows is None:
        return trace_data
    jittered = []
    for idx, row in enumerate(rows):
        if isinstance(row, (tuple, list)) and len(row) > COL_COMP_TIME:
            try:
                latency = float(row[COL_COMP_TIME])
            except (TypeError, ValueError):
                latency = None
            if latency is not None and latency > 0:
                factor = jitter_factor(seed, instance_id, batch_id, idx, sigma)
                row = tuple(row)
                row = (row[:COL_COMP_TIME]
                       + (str(max(1, int(round(latency * factor)))),)
                       + row[COL_COMP_TIME + 1:])
        jittered.append(row)
    trace_data.rows = jittered
    return trace_data


# ---------------------------------------------------------------------------
# audit output
# ---------------------------------------------------------------------------
def write_jsonl(path, records):
    """Write ground-truth records. Never called with a telemetry path."""
    with open(path, "w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")
    return len(records)
