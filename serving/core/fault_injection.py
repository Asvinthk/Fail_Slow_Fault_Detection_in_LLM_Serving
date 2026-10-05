"""Fail-slow fault injection for the PD-disaggregated simulator.

Prototype for college students: one small file, plain functions, no tricks.

Idea in one paragraph:
  A request flows prefill compute -> KV transfer -> decode compute.
  A fail-slow fault makes one stage slower but never raises an error.
  We model that by scaling numbers the simulator already produces
  (comp_time for compute, comm_size for transfer) for batches that
  belong to the faulty instance, only while the fault is active.

Timing rule (batch granularity):
  Faults are checked at schedule time using the sim clock `current`.
  A batch already sent to ASTRA-Sim is never modified.
  So onset can be late by up to one iteration. This is documented,
  not hidden.

These are SIMULATOR injection modes only. They say nothing about real
hardware causes (thermal, ECC, etc.). Synthetic faults cannot prove
detection of real hardware failures.
"""

import json

# Trace row columns produced by trace_generator.formatter.
# One layer row has 11 fields, see serving/core/utils.py.
COL_COMP_TIME = 1
COL_LAYER = 0
COL_COMM_SIZE = 9

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
                          "delay_ns", "link_bw_scale"}
    unknown = set(fault.keys()) - allowed
    if unknown:
        raise ValueError("fault %s has unknown keys: %s" % (fault.get("fault_id"), sorted(unknown)))
    if fault["fault_id"] in seen_ids:
        raise ValueError("duplicate fault_id: %s" % fault["fault_id"])
    target = fault["target"]
    if set(target.keys()) != {"role", "instance_id"}:
        raise ValueError("target must have exactly role and instance_id")
    if target["role"] not in ALLOWED_ROLES:
        raise ValueError("bad role: %s" % target["role"])
    if fault["mode"] not in ALLOWED_MODES:
        raise ValueError("bad mode: %s" % fault["mode"])
    if fault["mode"] in ("compute_multiplicative",) and "factor" not in fault:
        raise ValueError("compute_multiplicative needs 'factor'")
    if fault["mode"] in ("compute_additive",) and "delta_ns" not in fault:
        raise ValueError("compute_additive needs 'delta_ns'")
    if fault["mode"] in ("kv_transfer_bandwidth",) and "bandwidth_fraction" not in fault:
        raise ValueError("kv_transfer_bandwidth needs 'bandwidth_fraction'")
    if fault["mode"] in ("kv_handoff_delay",) and "delay_ns" not in fault:
        raise ValueError("kv_handoff_delay needs 'delay_ns'")
    if fault["mode"] in ("worker_stall",):
        if "factor" in fault or "delta_ns" in fault:
            raise ValueError("worker_stall takes no factor or delta_ns")
    schedule = fault["schedule"]
    if set(schedule.keys()) - {"start_s", "end_s", "profile", "period_s", "duty", "ramp_s"}:
        raise ValueError("schedule has unknown keys for %s" % fault["fault_id"])
    if "start_s" not in schedule or "end_s" not in schedule:
        raise ValueError("schedule needs start_s and end_s")
    if schedule["end_s"] <= schedule["start_s"]:
        raise ValueError("schedule end_s must be after start_s")
    profile = schedule.get("profile", "step")
    if profile not in ALLOWED_PROFILES:
        raise ValueError("bad profile: %s" % profile)
    if fault.get("injection_layer") == "output_approx" and fault["mode"] != "network_degradation":
        raise ValueError("output_approx is only allowed for network_degradation (latency)")


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
        if phase < duty * period:
            return 1.0
        return 0.0
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
        up = (now_s - start) / ramp
        down = (end - now_s) / ramp
        return max(0.0, min(1.0, up, down))
    return 1.0


def fault_matches(fault, role, instance_id):
    """True if this fault targets the given batch role and instance."""
    target = fault["target"]
    target_role = target["role"]
    target_inst = target["instance_id"]
    role_match = (target_role == role) or (target_role == "transfer" and role == "prefill")
    if not role_match:
        # network role with instance "all" hits every instance
        if target_role == "network" and target_inst == "all":
            return True
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


def apply_to_trace_rows(faults, rows, role, instance_id, now_s):
    """Scale comp_time and comm_size of trace rows. Returns audit entries.

    Called after generate_trace built the rows and before graph conversion,
    so the graph-cache digest changes naturally. Integer rounding is applied
    because the Chakra converter requires int comp_time.
    """
    audits = []
    for fault in faults:
        if fault["mode"] == "worker_stall":
            continue
        if fault["mode"] == "kv_handoff_delay":
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
        for row in rows:
            if len(row) != 11:
                continue
            layer = row[COL_LAYER]
            if mode in ("compute_multiplicative", "compute_additive", "network_degradation"):
                old = int(row[COL_COMP_TIME])
                nominal_comp += old
                if mode == "compute_multiplicative":
                    factor = 1.0 + weight * (fault["factor"] - 1.0)
                    new = int(round(old * factor))
                elif mode == "compute_additive":
                    new = int(old + round(weight * fault["delta_ns"]))
                else:
                    # network_degradation as output-level compute scaling placeholder
                    scale = 1.0 + weight * (fault.get("factor", 1.5) - 1.0)
                    new = int(round(old * scale))
                row[COL_COMP_TIME] = str(new)
                applied_comp += new
            if mode in ("kv_transfer_bandwidth", "network_degradation"):
                # Only qkv_proj rows carry the P/D KV transfer in comm_size.
                if layer.startswith("qkv_proj") and role == "prefill":
                    old_comm = int(row[COL_COMM_SIZE])
                    nominal_comm += old_comm
                    if mode == "kv_transfer_bandwidth":
                        frac = fault["bandwidth_fraction"]
                        new_comm = int(round(old_comm / frac))
                    else:
                        new_comm = int(round(old_comm * 1.0))
                    row[COL_COMM_SIZE] = str(new_comm)
                    applied_comm += new_comm
        # Additive per-iteration delay: documented as one extra serial delay.
        # The caller adds it to the last serial row if needed; here we keep
        # the per-layer math only, so students can see exactly what changed.
        audits.append({
            "fault_id": fault["fault_id"],
            "mode": mode,
            "weight": weight,
            "nominal_comp_sum": nominal_comp,
            "applied_comp_sum": applied_comp,
            "nominal_comm_sum": nominal_comm,
            "applied_comm_sum": applied_comm,
            "applied": True,
        })
    return audits
