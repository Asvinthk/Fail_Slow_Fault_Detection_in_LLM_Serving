"""Run the simulator, generate workloads, calibrate load, record manifests.

Producer side of the pipeline. Everything here answers "how do I obtain a
run and what was its operating point?"; nothing here reads ground truth
for scoring (that is evaluate.py) and nothing here decides what to inject
(that is inject_faults.py).

Contents
--------
workloads      seeded Poisson / bursty / sustained traces, plus the
               diurnal NHPP day profile. Arrivals and lengths are sampled
               independently of any fault parameter.
regime         lambda, U_prefill, U_decode, TPOT p50 measured from a
               finished run's telemetry. A detection number without its
               regime is not interpretable: a 1.25x fault is masked by
               queueing at 99% utilisation and diluted by idle time at 19%.
manifest       one JSONL line per run (faults, schedule, status, regime,
               git commit) plus the audit readers used for consistency.
separation     check_observable_clean: observable telemetry may not carry
               ground-truth columns OR values (a run_id of "..._faulty"
               separates the classes as cleanly as a column would).
simulator      run_simulator(): one `python -m serving` invocation with
               explicit cluster/workload/telemetry paths and a timeout.
calibration    calibrate_load(): bisect lambda until the decode pool sits
               at a target utilisation, then record the regime.

Why calibration exists
----------------------
Measured cost is ~127 s wall per heavy request, so a 24 h trace at 2 req/s
(~173k requests) is months of wall clock. The defensible unit is a
LOAD-MATCHED window: choose lambda so the decode pool sits at a stated
utilisation, run a few simulated minutes, and report that regime with every
number. Utilisation is fixed, severity is the independent variable.

Commands
--------
  python -m experiments.fault_detection.run_simulation workload \
      --pattern day --out w.jsonl --horizon-h 0.5 --seed 7
  python -m experiments.fault_detection.run_simulation calibrate \
      --cluster configs/cluster/pd_2p2d_small.json --target-u 0.75 \
      --out experiments/out/pilot_2p2d
  python -m experiments.fault_detection.run_simulation run \
      --cluster configs/cluster/pd_2p2d_small.json --workload w.jsonl \
      --out-csv outputs/x.csv --telemetry experiments/out/x_tel
"""

import argparse
import csv
import json
import math
import os
import random
import re
import subprocess
import sys

# ---------------------------------------------------------------------------
# observable / ground-truth separation
# ---------------------------------------------------------------------------
# Columns that must never appear in observable telemetry files.
MANIFEST_ONLY_COLUMNS = frozenset({
    "fault_present", "fault_id", "failure_class", "successfully_applied",
    "pair_id", "expected_label", "workload_id", "experiment_id",
})

# Substrings that must never appear in an observable VALUE. An identifier
# that spells out the label (e.g. run_id "..._faulty") separates the
# classes as cleanly as a ground-truth column would.
LEAKY_VALUE_TOKENS = ("fault", "healthy", "slow", "stall", "degrad",
                      "throttle", "neutral")

MODE_TO_FAILURE_CLASS = {
    "compute_multiplicative": "fail_slow",
    "compute_additive": "fail_slow",
    "kv_transfer_bandwidth": "fail_slow",
    "kv_handoff_delay": "fail_slow",
    "network_degradation": "fail_slow",
    "worker_stall": "fail_stop_stall",
    # Synthetic load-coupled throttling is a fail-slow mechanism too: the
    # component keeps serving, only its latency distribution degrades.
    "throttle": "fail_slow",
}

MODE_TO_COMPONENT = {
    "compute_multiplicative": None,  # resolved by target role below
    "compute_additive": None,
    "kv_transfer_bandwidth": "kv-transfer-network",
    "kv_handoff_delay": "kv-transfer-network",
    "network_degradation": "kv-transfer-network",
    "worker_stall": "worker-unavailable",
    "throttle": None,  # resolved by target role below
}


def expected_label_for_fault(fault):
    """Component an injected fault should localise to."""
    component = MODE_TO_COMPONENT.get(fault.get("mode"))
    if component is None:
        role = fault.get("target", {}).get("role")
        if role == "prefill":
            component = "prefill"
        elif role == "decode":
            component = "decode"
        elif role in ("transfer", "network"):
            component = "kv-transfer-network"
        else:
            component = "unknown"
    if fault.get("mode") == "worker_stall" \
            and fault.get("target", {}).get("role") in ("prefill", "decode"):
        # Stall localizes to the stalled role's pool; keep the
        # worker-unavailable class so it is not confused with fail-slow.
        component = "worker-unavailable"
    return component


def load_manifest(path):
    entries = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def load_audit(telemetry_dir):
    """Ground-truth records written next to telemetry, never inside it.

    Two files exist because the mechanisms are keyed differently:

      injection_audit.jsonl  per-batch trace scaling, handoff holds and
                             stall-window summaries (fault_id present);
      throttle_audit.jsonl   per-window load state and per-batch throttle
                             application (no fault_id -- the mechanism is
                             implied by the file).

    Throttle rows are tagged with fault_id ``throttle`` so the same
    consistency check and the same labelling path work for both. Neither
    file is ever merged into an observable CSV.
    """
    records = []
    for name, default_fault_id in (("injection_audit.jsonl", None),
                                   ("throttle_audit.jsonl", "throttle")):
        path = os.path.join(telemetry_dir, name)
        if not os.path.exists(path):
            continue
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                if default_fault_id is not None:
                    record.setdefault("fault_id", default_fault_id)
                records.append(record)
    return records


def check_observable_clean(telemetry_dir):
    """Raise ValueError if an observable file leaks ground truth.

    Two checks, both needed:

      * column names -- a manifest-only column must not exist;
      * cell values -- an identifier that spells out the label (the sim
        used to be invoked with ``--run-id <pair>_faulty``, which put the
        word "faulty" in every batch row) is just as much a leak as a
        column, and a detector reading run_id would trivially separate
        the classes. Callers must pass an opaque run id.
    """
    for name in ("batch_telemetry.csv", "request_telemetry.csv"):
        path = os.path.join(telemetry_dir, name)
        if not os.path.exists(path):
            continue
        with open(path) as f:
            reader = csv.reader(f)
            header = next(reader, [])
            leaked = [c for c in header if c in MANIFEST_ONLY_COLUMNS]
            if leaked:
                raise ValueError("observable file %s leaks ground truth columns: %s"
                                 % (path, leaked))
            for row in reader:
                for value in row:
                    low = value.lower()
                    hit = [t for t in LEAKY_VALUE_TOKENS if t in low]
                    if hit:
                        raise ValueError(
                            "observable file %s leaks a label through a value "
                            "(%r contains %s); pass an opaque run id"
                            % (path, value, hit))


def audit_within_interval(audit_records, fault):
    """True if an applied audit record for fault_id is inside [start, end)."""
    schedule = fault.get("schedule", {})
    start_ns = schedule.get("start_s", 0) * 1e9
    end_ns = schedule.get("end_s", float("inf")) * 1e9
    visited = False
    for record in audit_records:
        if record.get("fault_id") != fault.get("fault_id"):
            continue
        time_ns = record.get("time_ns", 0)
        if start_ns <= time_ns < end_ns:
            visited = True
            if record.get("applied"):
                return True, ""
    if visited:
        return False, ("records for %s are inside the window but applied=false "
                       "(mechanism neutral for those batches)" % fault.get("fault_id"))
    return False, "no audit record for %s inside [%s, %s)s" % (
        fault.get("fault_id"), schedule.get("start_s"), schedule.get("end_s"))


def check_audit_consistency(entry, audit_records):
    """Verify audit/manifest agreement for one manifest entry.

    Returns (ok, reason). Faulty runs need >=1 in-interval audit record
    per fault; healthy runs need empty audits.
    """
    faults = entry.get("faults", [])
    if not entry.get("fault_present"):
        if audit_records:
            return False, "healthy run has %d audit records" % len(audit_records)
        return True, ""
    for fault in faults:
        ok, reason = audit_within_interval(audit_records, fault)
        if not ok:
            return False, reason
    return True, ""


# ---------------------------------------------------------------------------
# workloads: poisson / bursty / sustained
# ---------------------------------------------------------------------------
def generate_workload(num, rate_sps, seed, short_in=32, long_in=512,
                      short_out=32, long_out=256, pattern="poisson",
                      off_s=1.0, on_s=1.0):
    """Seeded JSONL-ready rows: {input_toks, output_toks, arrival_time_ns}.

    Token ids are omitted on purpose, which disables prefix-cache hits and
    keeps control tight. Keep lengths inside the profiled sweep bounds from
    the profile meta.yaml (engine_effective); beyond them the simulator
    extrapolates and warns.
    """
    rnd = random.Random(seed)
    time_ns = 0
    rows = []
    if pattern == "sustained":
        gap_ns = int(1e9 / rate_sps) if rate_sps > 0 else 0
        gaps = [gap_ns] * num
    elif pattern == "bursty":
        gaps = []
        while len(gaps) < num:
            on_n = max(1, int(on_s * rate_sps))
            for _ in range(on_n):
                if len(gaps) >= num:
                    break
                gaps.append(int(rnd.expovariate(rate_sps) * 1e9) if rate_sps > 0 else 0)
            if len(gaps) < num:
                gaps[-1] += int(off_s * 1e9)
    else:  # poisson
        gaps = [(int(rnd.expovariate(rate_sps) * 1e9) if rate_sps > 0 else 0)
                for _ in range(num)]
    for gap_ns in gaps:
        time_ns += gap_ns
        input_toks = rnd.randint(short_in, long_in)
        output_toks = rnd.randint(short_out, long_out)
        rows.append({"input_toks": input_toks, "output_toks": output_toks,
                     "arrival_time_ns": time_ns})
    return rows


# ---------------------------------------------------------------------------
# workloads: diurnal NHPP day profile
# ---------------------------------------------------------------------------
DAY_DEFAULTS = {
    "horizon_s": 24 * 3600.0,
    "base_sps": 2.0,
    "peak_hour": 12.0,
    "peak_height": 3.0,
    "peak_sigma_h": 1.5,
    "afternoon_tau_h": 3.0,
    "evening_frac": 0.4,
    "bursts": [],
    "seed": 7,
    "short_in": 32,
    "long_in": 512,
    "short_out": 32,
    "long_out": 256,
}


def day_rate_at(t_s, cfg):
    """Instantaneous arrival rate (requests/s) at sim second t_s."""
    base = cfg["base_sps"]
    peak_h = cfg["peak_hour"] * 3600.0
    sigma = max(cfg["peak_sigma_h"] * 3600.0, 1.0)
    # Morning ramp: 0.2 -> 1.0 over [0.5h, peak - sigma].
    ramp_end = max(peak_h - sigma, 1800.0)
    if t_s < 1800.0:
        morning = 0.2
    elif t_s < ramp_end:
        morning = 0.2 + 0.8 * (t_s - 1800.0) / max(ramp_end - 1800.0, 1.0)
    else:
        morning = 1.0
    # Noon peak: Gaussian, configurable height (0 disables).
    noon = cfg["peak_height"] * math.exp(-0.5 * ((t_s - peak_h) / sigma) ** 2)
    # Afternoon decay: from peak level down to evening level.
    if t_s > peak_h:
        aft = math.exp(-(t_s - peak_h) / max(cfg["afternoon_tau_h"] * 3600.0, 1.0))
    else:
        aft = 1.0
    # Evening: flat fraction of base after 20:00.
    evening = cfg["evening_frac"] if t_s > 20 * 3600.0 else 1.0
    return max(base * (morning * (1.0 + noon) * aft) * evening, 0.0)


def intensity_at(t_s, cfg, bursts=None):
    """Full instantaneous arrival rate: diurnal shape + every burst term.

    This is the ONLY function that defines the process. Thinning accepts
    with ``intensity_at(t) / max_intensity(cfg)``, so the bound must be
    computed from this same function -- computing it from the peak and the
    bursts separately is not a bound when they overlap: at the default peak
    a burst reaches base*(1+peak_height) + base*(mult-1) = 10 req/s against
    an 8 req/s "bound", so rate/max > 1 and every candidate was accepted.
    """
    r = day_rate_at(t_s, cfg)
    for b in bursts or cfg.get("bursts", []):
        start = b.get("start_s")
        if start is None:
            continue
        if start <= t_s < start + b.get("dur_s", 60.0):
            r += cfg["base_sps"] * (b.get("rate_mult", 2.0) - 1.0)
    return max(r, 0.0)


def max_intensity(cfg, bursts=None, step_s=1.0):
    """Supremum of intensity_at over the horizon (grid + burst edges).

    A 1 s grid over 24 h is 86,400 evaluations -- cheap. Burst edges are
    evaluated exactly because the burst term is piecewise constant and a
    short burst can be narrower than the grid step.
    """
    horizon_s = cfg["horizon_s"]
    step = max(float(step_s), 0.1)
    best = 0.0
    t = 0.0
    while t < horizon_s:
        best = max(best, intensity_at(t, cfg, bursts))
        t += step
    for b in bursts or cfg.get("bursts", []):
        for edge in (b.get("start_s"),
                     b.get("start_s", 0.0) + b.get("dur_s", 60.0) - 1e-9):
            if edge is not None and 0.0 <= edge < horizon_s:
                best = max(best, intensity_at(edge, cfg, bursts))
    return max(best, 1e-9)


def expected_count(cfg, bursts=None, step_s=1.0):
    """Analytic expectation of the number of arrivals (integral of intensity)."""
    horizon_s = cfg["horizon_s"]
    step = max(float(step_s), 0.1)
    total = 0.0
    t = 0.0
    while t < horizon_s:
        total += intensity_at(t, cfg, bursts) * step
        t += step
    return total


def generate_day_workload(cfg):
    """Seeded NHPP day trace. Returns (rows, info).

    Known limitation (disclosed, not fixed): the profile is a ONE-SHOT day,
    not a periodic process. rate(23.99h) ~= 0.015 req/s against
    rate(0.1h) ~= 0.40 req/s, a ~27x step at midnight, because the
    afternoon decay is ~0.02 at the end of the day while the morning ramp
    restarts at 0.2. Concatenating two generated days is therefore not a
    steady-state continuation; a periodic model needs a wrapped harmonic
    basis. tests/test_run_simulation.py pins the discontinuity.
    """
    rnd = random.Random(cfg["seed"])
    horizon_s = cfg["horizon_s"]
    # Burst windows (seeded placement unless explicit start given). They
    # are resolved BEFORE the bound, because the bound depends on where
    # they land.
    bursts = []
    for b in cfg.get("bursts", []):
        start = b.get("start_s")
        if start is None:
            start = rnd.uniform(0, max(horizon_s - b.get("dur_s", 60.0), 0.0))
        bursts.append({"start_s": start, "dur_s": b.get("dur_s", 60.0),
                       "rate_mult": b.get("rate_mult", 2.0)})
    # Thinning: candidates at the true supremum of the full intensity,
    # accept with p = intensity/max.
    max_rate = max_intensity(cfg, bursts)
    t = 0.0
    arrivals = []
    while True:
        t += rnd.expovariate(max_rate)
        if t >= horizon_s:
            break
        r = intensity_at(t, cfg, bursts)
        if rnd.random() < r / max_rate:
            arrivals.append(t)
    rows = []
    for arrival in arrivals:
        rows.append({
            "input_toks": rnd.randint(cfg["short_in"], cfg["long_in"]),
            "output_toks": rnd.randint(cfg["short_out"], cfg["long_out"]),
            "arrival_time_ns": int(arrival * 1e9),
        })
    return rows, {"num_requests": len(rows), "horizon_s": horizon_s,
                  "bursts": bursts, "seed": cfg["seed"]}


def write_workload(path, rows):
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    return len(rows)


# ---------------------------------------------------------------------------
# regime: what load was this run actually at?
# ---------------------------------------------------------------------------
def _as_float(row, key, default=0.0):
    try:
        return float(row.get(key, default))
    except (TypeError, ValueError):
        return default


def _read_rows(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def _unique_batches(telemetry_dir):
    """One row per (instance, batch_id), phantom rows dropped.

    Regime measurement is producer-side: it runs before any evaluation
    code exists, so it deliberately does not import evaluate.py. The
    dedupe rule is the same one the feature builder uses (a TP=2 instance
    reports one batch twice; the smallest sys is the start NPU).
    """
    path = os.path.join(telemetry_dir, "batch_telemetry.csv")
    if not os.path.exists(path):
        return []
    best = {}
    for row in _read_rows(path):
        if _as_float(row, "batch_id") <= 0:
            continue
        key = (int(_as_float(row, "instance_id")), int(_as_float(row, "batch_id")))
        if key not in best or _as_float(row, "sys") < _as_float(best[key], "sys"):
            best[key] = row
    return [best[k] for k in sorted(best)]


def _instance_roles(telemetry_dir):
    path = os.path.join(telemetry_dir, "request_telemetry.csv")
    roles = {}
    if not os.path.exists(path):
        return roles
    for row in _read_rows(path):
        pre = _as_float(row, "prefill_instance_id", -1)
        dec = _as_float(row, "decode_instance_id", -1)
        if pre >= 0:
            roles[int(pre)] = "prefill"
        if dec >= 0:
            prev = roles.get(int(dec))
            roles[int(dec)] = "colocated" if prev == "prefill" else "decode"
    return roles


def regime_from_telemetry(telemetry_dir):
    """Return the regime dict for one run directory (never raises).

    Utilisation is busy time over the run span from DEDUPLICATED batch
    durations, so a TP=2 instance is not counted twice. It is a
    per-instance lower bound: overlapping batches are summed, then capped
    at 1.0.
    """
    empty = {"lambda_sps": -1.0, "u_prefill": -1.0, "u_decode": -1.0,
             "tpot_p50_ms": -1.0, "span_s": -1.0, "n_requests": 0}
    batches = _unique_batches(telemetry_dir)
    if not batches:
        return empty
    roles = _instance_roles(telemetry_dir)
    start = min(_as_float(r, "batch_time_ns") for r in batches)
    finish = max(_as_float(r, "finish_ns") for r in batches)
    span = (finish - start) / 1e9
    if span <= 0:
        return empty
    busy = {}
    for row in batches:
        inst = int(_as_float(row, "instance_id"))
        busy[inst] = busy.get(inst, 0.0) + _as_float(row, "duration_ns") / 1e9
    by_role = {}
    for inst, seconds in busy.items():
        by_role.setdefault(roles.get(inst, "unknown"), []).append(
            min(1.0, seconds / span))
    n_requests = 0
    tpot = -1.0
    req_path = os.path.join(telemetry_dir, "request_telemetry.csv")
    if os.path.exists(req_path):
        reqs = _read_rows(req_path)
        n_requests = len(reqs)
        tpots = [_as_float(r, "tpot", -1) for r in reqs]
        tpots = [v for v in tpots if v >= 0]
        tpot = (sorted(tpots)[len(tpots) // 2] / 1e6) if tpots else -1.0
    return {
        "lambda_sps": (n_requests / span) if n_requests else -1.0,
        "u_prefill": float(max(by_role.get("prefill", [0.0]))),
        "u_decode": float(max(by_role.get("decode", by_role.get("colocated", [0.0])))),
        "tpot_p50_ms": tpot,
        "span_s": span,
        "n_requests": n_requests,
    }


# ---------------------------------------------------------------------------
# one simulator run
# ---------------------------------------------------------------------------
def total_clocks_from_log(log_text):
    for line in log_text.splitlines():
        if "Total clocks (ns)" in line:
            digits = "".join(c for c in line if c.isdigit())
            if digits:
                return int(digits)
    return -1


def run_simulator(repo_root, cluster, workload_path, out_csv, telemetry_dir,
                  run_id, num_reqs=None, fault_path=None, inputs_root=None,
                  keep_inputs=False, timing_jitter=0.0, jitter_seed=1234,
                  timeout=1200, dtype="bfloat16", block_size=16,
                  throttle_path=None, extra_args=None):
    """Invoke `python -m serving` once. Returns a result dict, never raises.

    The simulator chdir's into astra-sim/, so every path passed here is
    made relative to repo_root (the sim resolves them against its own cwd
    after that chdir). --inputs-root is passed as-is because the sim
    abspaths it before the chdir.
    """
    if num_reqs is None:
        with open(workload_path) as f:
            num_reqs = sum(1 for _ in f)
    cmd = [sys.executable, "-m", "serving",
           "--cluster-config", cluster,
           "--dtype", dtype, "--block-size", str(block_size),
           "--dataset", os.path.relpath(workload_path, repo_root),
           "--output", os.path.relpath(out_csv, repo_root),
           "--num-reqs", str(num_reqs),
           "--run-id", run_id,
           "--telemetry-output", os.path.relpath(telemetry_dir, repo_root)]
    if inputs_root is not None:
        cmd += ["--inputs-root", inputs_root]
    if fault_path is not None:
        cmd += ["--fault-config", os.path.relpath(fault_path, repo_root)]
    if throttle_path is not None:
        cmd += ["--throttle-config", os.path.relpath(throttle_path, repo_root)]
    if keep_inputs:
        cmd += ["--keep-inputs"]
    if timing_jitter and timing_jitter > 0:
        cmd += ["--timing-jitter", str(timing_jitter),
                "--timing-jitter-seed", str(jitter_seed)]
    if extra_args:
        cmd += list(extra_args)
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              cwd=repo_root, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"status": "failed", "reason": "timeout", "clocks": -1,
                "log": "TIMEOUT after %ds\n" % timeout, "returncode": None,
                "cmd": cmd}
    log_text = proc.stdout + proc.stderr
    status = "ok" if proc.returncode == 0 else "failed"
    reason = "" if proc.returncode == 0 else "exit %d" % proc.returncode
    return {"status": status, "reason": reason,
            "clocks": total_clocks_from_log(log_text) if status == "ok" else -1,
            "log": log_text, "returncode": proc.returncode, "cmd": cmd}


# ---------------------------------------------------------------------------
# calibration: lambda for a target decode utilisation
# ---------------------------------------------------------------------------
def probe_run_id(out_root, tag):
    """Unique, path-safe simulator run id for one calibration probe.

    The simulator writes its ASTRA-Sim input tree under
    ``astra-sim/inputs/runs/<run_id>``, a path shared by every invocation.
    A bare ``probe_00`` therefore collides between two calibrations (two
    targets, or the same target re-run) and lets one run overwrite the
    other's workload/system files *while they are being read* -- which
    silently mixes two workloads into one trace. Including the out-root
    basename keeps concurrent calibrations disjoint.
    """
    base = os.path.basename(os.path.abspath(out_root)) or "cal"
    return re.sub(r"[^A-Za-z0-9_.-]", "_", "%s_%s" % (base, tag))


def calibrate_load(repo_root, out_root, cluster, target_u, num_reqs=40,
                   iters=5, tol=0.05, seed=11, timeout=1800,
                   timing_jitter=0.0, short_in=32, long_in=512,
                   short_out=32, long_out=256, lambda_lo=0.05,
                   lambda_hi=64.0):
    """Bisect lambda until U_decode is near target_u. Returns a record.

    Geometric midpoint because arrival rates span decades. Each probe is a
    real seeded Poisson run with telemetry; the probe closest to the target
    is reported as the chosen regime.

    ``lambda_lo``/``lambda_hi`` bracket the search. The default spans two
    decades, which is wasteful when a previous calibration already
    localised the operating point: near saturation U is flat in lambda, so
    the upper probes return U=1.0 and teach nothing.
    """
    lo, hi = lambda_lo, lambda_hi
    probes = []
    for i in range(iters):
        lam = (lo * hi) ** 0.5
        tag = "probe_%02d" % i
        probe_dir = os.path.join(out_root, tag)
        os.makedirs(probe_dir, exist_ok=True)
        workload_path = os.path.join(probe_dir, "workload.jsonl")
        rows = generate_workload(num=num_reqs, rate_sps=lam, seed=seed,
                                 short_in=short_in, long_in=long_in,
                                 short_out=short_out, long_out=long_out)
        write_workload(workload_path, rows)
        telemetry_dir = os.path.join(probe_dir, "telemetry")
        os.makedirs(telemetry_dir, exist_ok=True)
        result = run_simulator(
            repo_root, cluster, workload_path,
            os.path.join(probe_dir, "sim.csv"), telemetry_dir,
            probe_run_id(out_root, tag),
            num_reqs=len(rows), timeout=timeout, timing_jitter=timing_jitter)
        if result["status"] != "ok":
            raise RuntimeError("probe %s failed: %s"
                               % (tag, result["log"][-2000:]))
        regime = regime_from_telemetry(telemetry_dir)
        regime.update({"lambda_requested": lam, "tag": tag})
        probes.append(regime)
        print("probe %d: lambda=%.3f/s -> U_decode=%.3f U_prefill=%.3f "
              "TPOT=%.2fms span=%.1fs"
              % (i, lam, regime["u_decode"], regime["u_prefill"],
                 regime["tpot_p50_ms"], regime["span_s"]))
        if abs(regime["u_decode"] - target_u) <= tol:
            break
        if regime["u_decode"] < target_u:
            lo = lam
        else:
            hi = lam
    best = min(probes, key=lambda r: abs(r["u_decode"] - target_u))
    return {
        "cluster": cluster,
        "target_u": target_u,
        "chosen": best,
        "probes": probes,
        "note": ("Load-matched pilot: lambda chosen for the decode-pool "
                 "utilisation above. Report this regime with every detection "
                 "number measured on this cluster."),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _day_config_from_args(args):
    cfg = dict(DAY_DEFAULTS)
    cfg.update({
        "horizon_s": args.horizon_h * 3600.0,
        "base_sps": args.base_sps,
        "peak_hour": args.peak_hour,
        "peak_height": args.peak_height,
        "peak_sigma_h": args.peak_sigma_h,
        "afternoon_tau_h": args.afternoon_tau_h,
        "evening_frac": args.evening_frac,
        "seed": args.seed,
        "short_in": args.short_in,
        "long_in": args.long_in,
        "short_out": args.short_out,
        "long_out": args.long_out,
    })
    bursts = [{"dur_s": args.burst_dur_s, "rate_mult": args.burst_rate_mult}
              for _ in range(max(args.bursts, 0))]
    cfg["bursts"] = bursts
    return cfg


def _cmd_workload(args):
    if args.pattern == "day":
        cfg = _day_config_from_args(args)
        rows, info = generate_day_workload(cfg)
        info.update({k: cfg[k] for k in ("base_sps", "peak_hour", "peak_height",
                                         "peak_sigma_h", "afternoon_tau_h",
                                         "evening_frac")})
    else:
        rows = generate_workload(
            num=args.num, rate_sps=args.rate_sps, seed=args.seed,
            short_in=args.short_in, long_in=args.long_in,
            short_out=args.short_out, long_out=args.long_out,
            pattern=args.pattern, off_s=args.off_s, on_s=args.on_s)
        info = {"num_requests": len(rows), "pattern": args.pattern,
                "rate_sps": args.rate_sps, "seed": args.seed}
    write_workload(args.out, rows)
    if args.manifest:
        with open(args.manifest, "w") as f:
            json.dump(info, f, indent=2)
    print("wrote %d rows to %s" % (len(rows), args.out))
    return 0


def _cmd_calibrate(args):
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    out_root = args.out if os.path.isabs(args.out) else os.path.join(repo_root, args.out)
    os.makedirs(out_root, exist_ok=True)
    record = calibrate_load(repo_root, out_root, args.cluster, args.target_u,
                            num_reqs=args.num_reqs, iters=args.iters,
                            tol=args.tol, seed=args.seed, timeout=args.timeout,
                            timing_jitter=args.timing_jitter,
                            short_in=args.short_in, long_in=args.long_in,
                            short_out=args.short_out, long_out=args.long_out,
                            lambda_lo=args.lambda_lo, lambda_hi=args.lambda_hi)
    path = os.path.join(out_root, "regime.json")
    with open(path, "w") as f:
        json.dump(record, f, indent=2)
    chosen = record["chosen"]
    print("chosen lambda=%.3f/s (U_decode=%.3f, target %.2f) -> %s"
          % (chosen["lambda_requested"], chosen["u_decode"],
             record["target_u"], path))
    return 0


def _cmd_run(args):
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    telemetry_dir = args.telemetry
    os.makedirs(telemetry_dir, exist_ok=True)
    result = run_simulator(repo_root, args.cluster, args.workload,
                           args.out_csv, telemetry_dir, args.run_id,
                           fault_path=args.fault_config,
                           timing_jitter=args.timing_jitter,
                           timeout=args.timeout)
    with open(args.out_csv.replace(".csv", ".log"), "w") as f:
        f.write(result["log"])
    if result["status"] != "ok":
        print("run failed: %s" % result["reason"])
        return 1
    regime = regime_from_telemetry(telemetry_dir)
    print("Total clocks (ns): %d  regime: lambda=%.3f/s U_decode=%.3f "
          "U_prefill=%.3f TPOT=%.2fms"
          % (result["clocks"], regime["lambda_sps"], regime["u_decode"],
             regime["u_prefill"], regime["tpot_p50_ms"]))
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="command", required=True)

    w = sub.add_parser("workload", help="generate a workload JSONL")
    w.add_argument("--out", required=True)
    w.add_argument("--pattern", default="poisson",
                   choices=["poisson", "bursty", "sustained", "day"])
    w.add_argument("--num", type=int, default=20)
    w.add_argument("--rate-sps", type=float, default=2.0)
    w.add_argument("--seed", type=int, default=42)
    w.add_argument("--short-in", type=int, default=32)
    w.add_argument("--long-in", type=int, default=512)
    w.add_argument("--short-out", type=int, default=32)
    w.add_argument("--long-out", type=int, default=256)
    w.add_argument("--on-s", type=float, default=1.0)
    w.add_argument("--off-s", type=float, default=1.0)
    w.add_argument("--horizon-h", type=float, default=24.0)
    w.add_argument("--base-sps", type=float, default=2.0)
    w.add_argument("--peak-hour", type=float, default=12.0)
    w.add_argument("--peak-height", type=float, default=3.0)
    w.add_argument("--peak-sigma-h", type=float, default=1.5)
    w.add_argument("--afternoon-tau-h", type=float, default=3.0)
    w.add_argument("--evening-frac", type=float, default=0.4)
    w.add_argument("--bursts", type=int, default=0)
    w.add_argument("--burst-dur-s", type=float, default=60.0)
    w.add_argument("--burst-rate-mult", type=float, default=2.0)
    w.add_argument("--manifest", default=None)
    w.set_defaults(func=_cmd_workload)

    c = sub.add_parser("calibrate", help="find lambda for a target decode utilisation")
    c.add_argument("--cluster", required=True)
    c.add_argument("--target-u", type=float, default=0.75)
    c.add_argument("--out", required=True)
    c.add_argument("--num-reqs", type=int, default=40)
    c.add_argument("--iters", type=int, default=5)
    c.add_argument("--tol", type=float, default=0.05)
    c.add_argument("--seed", type=int, default=11)
    c.add_argument("--timeout", type=int, default=1800)
    c.add_argument("--timing-jitter", type=float, default=0.0)
    c.add_argument("--short-in", type=int, default=32)
    c.add_argument("--long-in", type=int, default=512)
    c.add_argument("--short-out", type=int, default=32)
    c.add_argument("--long-out", type=int, default=256)
    c.add_argument("--lambda-lo", type=float, default=0.05,
                   help="lower bracket for the arrival-rate bisection")
    c.add_argument("--lambda-hi", type=float, default=64.0,
                   help="upper bracket; narrow it once a previous "
                        "calibration has localised the operating point")
    c.set_defaults(func=_cmd_calibrate)

    r = sub.add_parser("run", help="one simulator run with telemetry")
    r.add_argument("--cluster", required=True)
    r.add_argument("--workload", required=True)
    r.add_argument("--out-csv", required=True)
    r.add_argument("--telemetry", required=True)
    r.add_argument("--run-id", default="run")
    r.add_argument("--fault-config", default=None)
    r.add_argument("--timing-jitter", type=float, default=0.0)
    r.add_argument("--timeout", type=int, default=1800)
    r.set_defaults(func=_cmd_run)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
