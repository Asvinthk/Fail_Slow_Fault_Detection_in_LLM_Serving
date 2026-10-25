"""Configure and execute paired healthy/faulty runs for all fault families.

Given a matrix profile, each (workload, cluster, seed) becomes ONE pair: a
single workload file is generated once and replayed by a healthy twin and
one faulty twin per fault variant. Same-trace pairing is mandatory -- a
"workload-only negative" is the healthy run on the identical trace, so a
difference between the twins cannot be a workload artefact.

Fault families come from serving/core/fault_study.py:
  compute_multiplicative / compute_additive   compute-side fail-slow
  kv_transfer_bandwidth / kv_handoff_delay    transfer-side fail-slow
  network_degradation                         collective-side fail-slow
  worker_stall                                fail-stop (contrast class)
plus synthetic load-coupled throttling and benign timing jitter, both
enabled from the matrix rather than hard-coded.

Severity ladders apply to the fault's own numeric knob (factor,
bandwidth_fraction or delay_ns), so a bandwidth sweep and a severity sweep
are the same mechanism.

Commands
--------
  python -m experiments.fault_detection.inject_faults \
      --matrix experiments/fault_detection/matrix.yaml \
      --profile review_min --out experiments/out/review_min
  python -m experiments.fault_detection.inject_faults ... --seeds 1,2,3,4,5

Resumable: pairs already "ok" in the manifest are skipped. Generated data
stays gitignored (experiments/out/).
"""

import argparse
import concurrent.futures
import copy
import hashlib
import json
import os
import subprocess
import sys
import threading

import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from experiments.fault_detection.run_simulation import (  # noqa: E402
    MODE_TO_FAILURE_CLASS,
    check_audit_consistency,
    check_observable_clean,
    expected_label_for_fault,
    generate_day_workload,
    generate_workload,
    load_audit,
    load_manifest,
    regime_from_telemetry,
    run_simulator,
    write_workload,
)

SEVERITY_KNOBS = ("factor", "bandwidth_fraction", "delay_ns", "slowdown_factor")


# ---------------------------------------------------------------------------
# matrix expansion
# ---------------------------------------------------------------------------
def apply_severity(fault, severity):
    """Set the fault's own numeric knob to a ladder value."""
    for knob in SEVERITY_KNOBS:
        if knob in fault:
            fault[knob] = float(severity) if knob != "delay_ns" else int(float(severity))
            return fault
    raise ValueError("severity ladder on a fault with no numeric knob: %s"
                     % fault.get("fault_id"))


def resolve_fault(templates, variant):
    """Template name (optionally 'name@severity') -> concrete fault dict."""
    if "@" in variant:
        name, severity = variant.split("@", 1)
        fault = copy.deepcopy(templates[name])
        return apply_severity(fault, severity)
    return copy.deepcopy(templates[variant])


def expand_profile(matrix, profile_name):
    """Flatten a profile into jobs: one per (cluster, pair, fault variant)."""
    profile = matrix["profiles"][profile_name]
    workloads = matrix["workloads"]
    clusters = matrix["clusters"]
    templates = matrix.get("fault_templates", {})
    jobs = []
    for cluster_id in profile["clusters"]:
        for pair in profile["pairs"]:
            workload_id = pair["workload"]
            fault_name = pair.get("fault")
            ladder = pair.get("severity_ladder")
            variants = [fault_name]
            if ladder and fault_name:
                variants = ["%s@%s" % (fault_name, s) for s in ladder]
            for variant in variants:
                jobs.append({
                    "workload_id": workload_id,
                    "workload": workloads[workload_id],
                    "cluster_id": cluster_id,
                    "cluster": clusters[cluster_id],
                    "fault_name": variant,
                    "fault": None if variant is None else resolve_fault(
                        templates, variant),
                })
    return jobs


def resolve_instance_ids(fault, cluster_path, repo_root):
    """Resolve target instance_id -1 to the last decode instance of a cluster."""
    with open(os.path.join(repo_root, cluster_path)) as f:
        cluster = json.load(f)
    instances = []
    for node in cluster.get("nodes", []):
        instances.extend(node.get("instances", []))
    decodes = [i for i, inst in enumerate(instances)
               if inst.get("pd_type") == "decode"]
    fault = copy.deepcopy(fault)
    target = fault.get("target", {})
    if target.get("instance_id") == -1 and decodes:
        target["instance_id"] = decodes[-1]
    return fault


def throttle_gpus_for_instance(cluster_path, repo_root, instance_id):
    """Physical NPU ids owned by one instance, in cluster-config order.

    Mirrors the simulator's allocation rule (config_builder): a prefill
    instance reserves ``num_npus`` NPUs plus the same number again as the
    shadow KV receiver range, while only the first range is real hardware.
    Throttle configs may only name real NPUs, so the shadow range is
    skipped here too.
    """
    with open(os.path.join(repo_root, cluster_path)) as f:
        cluster = json.load(f)
    start = 0
    idx = 0
    for node in cluster.get("nodes", []):
        for inst in node.get("instances", []):
            n = int(inst.get("tp_size", 1)) * int(inst.get("pp_size", 1))
            if idx == instance_id:
                return list(range(start, start + n))
            start += 2 * n if inst.get("pd_type") == "prefill" else n
            idx += 1
    return []


def git_commit(repo_root):
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, cwd=repo_root,
                             timeout=30)
        return out.stdout.strip() if out.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def _build_workload(job, seed, path):
    params = dict(job["workload"])
    params["seed"] = seed
    if params.get("pattern") == "day":
        # Day-profile rows are produced by the NHPP generator; its config
        # uses horizon/base/peak keys rather than num/rate.
        params.pop("pattern")
        rows, _info = generate_day_workload(params)
    else:
        # `pattern` must stay in params: dropping it silently turned every
        # bursty workload into the Poisson default (caught by the smoke
        # clock comparison, pinned by a unit test).
        rows = generate_workload(**params)
    return write_workload(path, rows)


# ---------------------------------------------------------------------------
# one pair: healthy twin + one faulty twin per fault variant
# ---------------------------------------------------------------------------
def run_pair(repo_root, out_root, pair_id, job, faults, seed, timeout,
             keep_inputs, commit, profile_name, timing_jitter=0.0,
             jitter_seed=1234, manifest_file=None, manifest_lock=None):
    workload_id = job["workload_id"]
    cluster_id = job["cluster_id"]
    pair_dir = os.path.join(out_root, pair_id)
    os.makedirs(pair_dir, exist_ok=True)

    # One workload file shared by the healthy and faulty twins.
    workload_path = os.path.join(pair_dir, "workload.jsonl")
    if not os.path.exists(workload_path):
        _build_workload(job, seed, workload_path)

    results = []
    twins = [("healthy", None)]
    for i, fault_spec in enumerate(faults):
        twins.append(("faulty%d" % i if len(faults) > 1 else "faulty",
                      fault_spec))
    for suffix, fault_spec in twins:
        run_id = "%s_%s" % (pair_id, suffix)
        # The sim writes --run-id into every observable batch row. A
        # readable id ("<pair>_faulty") leaks the label through a VALUE,
        # which check_observable_clean rejects. The manifest keeps the
        # readable id; the sim gets an opaque token.
        telemetry_run_id = "run_%s" % hashlib.sha256(
            ("%s|%s|%d" % (pair_id, suffix, seed)).encode()).hexdigest()[:12]
        telemetry_dir = os.path.join(pair_dir, "%s_telemetry" % suffix)
        os.makedirs(telemetry_dir, exist_ok=True)
        csv_path = os.path.join(pair_dir, "%s.csv" % suffix)
        log_path = os.path.join(pair_dir, "%s.log" % suffix)
        fault_path = None
        throttle_path = None
        resolved = None
        if fault_spec is not None:
            resolved = resolve_instance_ids(fault_spec, job["cluster"], repo_root)
            if resolved.get("mode") == "throttle":
                # Synthetic load-coupled throttling is configured per
                # physical GPU, not by a fault schedule: build the throttle
                # config for the target instance's NPUs.
                gpus = throttle_gpus_for_instance(
                    job["cluster"], repo_root,
                    resolved["target"]["instance_id"])
                if not gpus:
                    raise ValueError("no NPUs resolved for throttle target %s"
                                     % resolved["target"])
                throttle_path = os.path.join(pair_dir, "throttle_%s.json" % suffix)
                with open(throttle_path, "w") as f:
                    json.dump({"gpus": [
                        {"gpu_id": g,
                         "tau_s": resolved.get("tau_s", 600.0),
                         "alpha": resolved.get("alpha", 1.0),
                         "th_on": resolved.get("th_on", 0.75),
                         "th_off": resolved.get("th_off", 0.60),
                         "slowdown_factor": resolved["slowdown_factor"],
                         "m_min": resolved.get("m_min", 0.5),
                         "window_s": resolved.get("window_s", 60.0)}
                        for g in gpus]}, f, indent=2)
            else:
                fault_path = os.path.join(pair_dir, "fault_%s.json" % suffix)
                with open(fault_path, "w") as f:
                    json.dump({"faults": [resolved]}, f, indent=2)
        # serving chdir's into astra-sim/, so repo-relative paths need the
        # same ".." prefix as the dataset/cluster flags (--inputs-root is
        # resolved with os.path.abspath against the sim cwd).
        inputs_rel = os.path.join("experiments", "out", pair_id,
                                  "inputs_%s" % suffix)
        if os.path.isabs(out_root):
            inputs_arg = os.path.join(out_root, pair_id, "inputs_%s" % suffix)
        else:
            inputs_arg = os.path.join("..", inputs_rel)
        result = run_simulator(
            repo_root, job["cluster"], workload_path, csv_path,
            telemetry_dir, telemetry_run_id, fault_path=fault_path,
            inputs_root=inputs_arg, keep_inputs=keep_inputs,
            timing_jitter=timing_jitter, jitter_seed=jitter_seed,
            timeout=timeout, throttle_path=throttle_path)
        with open(log_path, "w") as f:
            f.write(result["log"])
        status, reason, clocks = result["status"], result["reason"], result["clocks"]
        regime = regime_from_telemetry(telemetry_dir)
        if resolved is not None and resolved.get("mode") == "throttle" \
                and regime.get("span_s", -1) > 0:
            # Throttling has no injected schedule: it is on whenever load
            # keeps the GPU hot. Pin the manifest schedule to the observed
            # span so "recovering" windows stay a real post-run horizon
            # instead of covering the whole trace.
            resolved["schedule"] = {"start_s": 0.0,
                                    "end_s": float(regime["span_s"]),
                                    "profile": "step"}

        audit = load_audit(telemetry_dir)
        try:
            check_observable_clean(telemetry_dir)
            leak = ""
        except ValueError as exc:
            leak = str(exc)
            status, reason = "failed", "telemetry leak: %s" % exc
        faults_here = [resolved] if (resolved is not None and suffix != "healthy") else []
        for flt in faults_here:
            flt["failure_class"] = MODE_TO_FAILURE_CLASS.get(
                flt.get("mode"), "fail_slow")
        if faults_here:
            ok, consistency = check_audit_consistency(
                {"fault_present": True, "faults": faults_here}, audit)
            applied = ok
            if not ok and status == "ok":
                reason = "audit inconsistency: %s" % consistency
        else:
            applied = False
            if audit:
                status, reason = "failed", "healthy run has audit records"
        entry = {
            "experiment_id": "fi_%s" % profile_name,
            "run_id": run_id,
            "pair_id": pair_id,
            # Directory name is recorded rather than re-derived: a pair
            # with several fault variants writes faulty0_telemetry,
            # faulty1_telemetry, ... not a single faulty_telemetry, and
            # evaluate.py must not have to guess which twin is which.
            "telemetry_dir": "%s_telemetry" % suffix,
            "workload_id": workload_id,
            "workload_params": job["workload"],
            "cluster_config_id": cluster_id,
            "seed": seed,
            "git_commit": commit,
            "fault_present": suffix != "healthy" and bool(faults_here),
            "faults": faults_here,
            "failure_class": faults_here[0]["failure_class"] if faults_here else "",
            "successfully_applied": applied,
            "expected_label": (expected_label_for_fault(faults_here[0])
                               if faults_here else "healthy"),
            "status": status,
            "reason": reason + ("; %s" % leak if leak else ""),
            "total_clocks_ns": clocks,
            # Benign-noise level is part of the operating point: a
            # detectability number at sigma=0.05 is not comparable to one
            # at sigma=0.
            "timing_jitter_sigma": timing_jitter,
            # Regime: the load a number was measured at. A detection
            # result is meaningless without it.
            "regime": regime,
        }
        if manifest_file is not None:
            # Flush per RUN, not per pair. A pair is 1-12 simulator runs
            # (20-40 min); losing a pair's manifest lines to one bad run
            # means re-simulating work that already succeeded.
            with manifest_lock:
                manifest_file.write(json.dumps(entry) + "\n")
                manifest_file.flush()
        results.append(entry)
    return results


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--matrix", required=True)
    parser.add_argument("--profile", default="smoke")
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=42,
                        help="single seed (kept for backwards compatibility)")
    parser.add_argument("--seeds", default=None,
                        help="comma-separated seeds; overrides --seed and runs "
                             "one pair set per seed")
    parser.add_argument("--timeout", type=int, default=1200)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--keep-inputs", action="store_true", default=False)
    parser.add_argument("--timing-jitter", type=float, default=None,
                        help="benign lognormal noise on compute time (0 = off, "
                             "bit-identical). Default: the profile's "
                             "noise_sigmas list from matrix.yaml.")
    parser.add_argument("--jitter-seed", type=int, default=1234)
    args = parser.parse_args(argv)

    seeds = ([int(s) for s in args.seeds.split(",") if s.strip()]
             if args.seeds else [args.seed])
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    with open(args.matrix) as f:
        matrix = yaml.safe_load(f)
    jobs = expand_profile(matrix, args.profile)
    profile = matrix["profiles"][args.profile]
    # Benign noise is part of the operating point, so it belongs in the
    # matrix: every (seed, sigma) combination gets its own pair_id and its
    # own healthy twin. A CLI value overrides the list (single-run use).
    if args.timing_jitter is not None:
        sigmas = [float(args.timing_jitter)]
    else:
        sigmas = [float(s) for s in profile.get("noise_sigmas", [0.0])]
    out_root = args.out if os.path.isabs(args.out) else os.path.join(repo_root, args.out)
    os.makedirs(out_root, exist_ok=True)
    manifest_path = os.path.join(out_root, "manifest.jsonl")

    done_pairs = set()
    if os.path.exists(manifest_path):
        for entry in load_manifest(manifest_path):
            if entry.get("status") == "ok":
                done_pairs.add((entry["pair_id"], entry["run_id"]))

    commit = git_commit(repo_root)
    # Group jobs by pair so the healthy twin runs once even when several
    # matrix entries share a (workload, cluster, seed) pair.
    pairs = {}
    for sigma in sigmas:
        for seed in seeds:
            for job in jobs:
                pair_id = "%s_%s_%d" % (job["workload_id"], job["cluster_id"], seed)
                if sigma > 0:
                    pair_id += "_n%s" % ("%g" % sigma)
                entry = pairs.setdefault((pair_id, seed),
                                         {"job": job, "faults": [],
                                          "sigma": sigma})
                if job["fault"] is not None:
                    entry["faults"].append(job["fault"])
    pending = []
    for (pair_id, seed), entry in pairs.items():
        expected = ["healthy"] + (["faulty%d" % i for i in range(len(entry["faults"]))]
                                  if len(entry["faults"]) > 1
                                  else ["faulty"] if entry["faults"] else [])
        if all((pair_id, "%s_%s" % (pair_id, suffix)) in done_pairs
               for suffix in expected):
            continue
        pending.append((pair_id, seed, entry))
    print("profile %s: %d pairs x %d seed(s) x %d noise level(s) (%d pending)"
          % (args.profile, len(pairs), len(seeds), len(sigmas), len(pending)))

    manifest_file = open(manifest_path, "a")
    manifest_lock = threading.Lock()
    failures = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        future_to_job = {
            pool.submit(run_pair, repo_root, out_root, pair_id,
                        entry["job"], entry["faults"], seed, args.timeout,
                        args.keep_inputs, commit, args.profile,
                        entry["sigma"], args.jitter_seed,
                        manifest_file, manifest_lock): pair_id
            for pair_id, seed, entry in pending}
        for future in concurrent.futures.as_completed(future_to_job):
            try:
                entries = future.result()
            except Exception as exc:  # never lose the manifest line
                print("JOB FAILED: %s" % exc)
                failures += 1
                continue
            # run_pair already appended each entry (flushed) under the lock.
            for entry in entries:
                print("%s %s clocks=%s applied=%s U_dec=%.3f" % (
                    entry["run_id"], entry["status"],
                    entry["total_clocks_ns"], entry["successfully_applied"],
                    entry["regime"]["u_decode"]))
                if entry["status"] != "ok":
                    failures += 1
    manifest_file.close()
    print("done, %d failed run(s)" % failures)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
