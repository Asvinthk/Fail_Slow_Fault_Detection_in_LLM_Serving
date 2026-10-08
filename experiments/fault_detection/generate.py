"""Paired experiment runner for fail-slow fault-detection research.

Single command:
  python -m experiments.fault_detection.generate --matrix experiments/fault_detection/matrix.yaml --profile smoke --out experiments/out/

For each (workload, cluster, seed): generate one workload file, then run
healthy then faulty with identical arrivals and lengths. The two runs
share a pair_id. Writes manifest.jsonl (one line per run), per-run
telemetry dirs, and fault JSONs. Resumable (skips pair_ids already in
the manifest), per-run timeout, parallel workers with distinct
--run-ids and --inputs-roots; run dirs auto-clean unless
--keep-inputs.

Generated data stays gitignored (experiments/out/).
"""

import argparse
import concurrent.futures
import copy
import json
import os
import subprocess
import sys
import time

import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from experiments.fault_detection.workloads import generate as generate_workload  # noqa: E402
from experiments.fault_detection.manifest import (  # noqa: E402
    MODE_TO_FAILURE_CLASS,
    check_audit_consistency,
    check_observable_clean,
    expected_label_for_fault,
    load_audit,
    load_manifest,
)


def git_commit(repo_root):
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, cwd=repo_root,
                             timeout=30)
        return out.stdout.strip() if out.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def expand_profile(matrix, profile_name):
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
            fault_variants = [fault_name]
            if ladder and fault_name:
                fault_variants = ["%s@%s" % (fault_name, s) for s in ladder]
            for variant in fault_variants:
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


def resolve_fault(templates, variant):
    if "@" in variant:
        name, severity = variant.split("@", 1)
        fault = copy.deepcopy(templates[name])
        if "factor" in fault:
            fault["factor"] = float(severity)
        return fault
    fault = copy.deepcopy(templates[variant])
    # instance_id -1 means "last decode instance"; resolved per cluster below.
    return fault


def resolve_instance_ids(fault, cluster_path, repo_root):
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


def total_clocks_from_log(log_text):
    for line in log_text.splitlines():
        if "Total clocks (ns)" in line:
            digits = "".join(c for c in line if c.isdigit())
            if digits:
                return int(digits)
    return -1


def run_one(repo_root, out_root, pair_id, job, faults, seed, timeout,
            keep_inputs, commit):
    workload_id = job["workload_id"]
    cluster_id = job["cluster_id"]
    pair_dir = os.path.join(out_root, pair_id)
    os.makedirs(pair_dir, exist_ok=True)

    # One workload file shared by the healthy and faulty twins.
    workload_path = os.path.join(pair_dir, "workload.jsonl")
    if not os.path.exists(workload_path):
        workload_params = dict(job["workload"])
        workload_params["seed"] = seed
        rows = generate_workload(**workload_params)
        with open(workload_path, "w") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")

    results = []
    twins = [("healthy", None)]
    for i, fault_spec in enumerate(faults):
        twins.append(("faulty%d" % i if len(faults) > 1 else "faulty",
                      fault_spec))
    for suffix, fault_spec in twins:
        run_id = "%s_%s" % (pair_id, suffix)
        telemetry_dir = os.path.join(pair_dir, "%s_telemetry" % suffix)
        os.makedirs(telemetry_dir, exist_ok=True)
        csv_path = os.path.join(pair_dir, "%s.csv" % suffix)
        log_path = os.path.join(pair_dir, "%s.log" % suffix)
        fault_path = None
        if fault_spec is not None:
            resolved = resolve_instance_ids(fault_spec, job["cluster"], repo_root)
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
        cmd = [sys.executable, "-m", "serving",
               "--cluster-config", job["cluster"],
               "--dtype", "bfloat16", "--block-size", "16",
               "--dataset", os.path.relpath(workload_path, repo_root),
               "--output", os.path.relpath(csv_path, repo_root),
               "--num-reqs", str(len(open(workload_path).readlines())),
               "--run-id", run_id,
               "--inputs-root", inputs_arg,
               "--telemetry-output", os.path.relpath(telemetry_dir, repo_root)]
        if fault_path is not None:
            cmd += ["--fault-config", os.path.relpath(fault_path, repo_root)]
        if keep_inputs:
            cmd += ["--keep-inputs"]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True,
                                  cwd=repo_root, timeout=timeout)
            log_text = proc.stdout + proc.stderr
            with open(log_path, "w") as f:
                f.write(log_text)
            status = "ok" if proc.returncode == 0 else "failed"
            reason = "" if proc.returncode == 0 else "exit %d" % proc.returncode
            clocks = total_clocks_from_log(log_text) if status == "ok" else -1
        except subprocess.TimeoutExpired:
            status, reason, clocks = "failed", "timeout", -1
            with open(log_path, "w") as f:
                f.write("TIMEOUT after %ds\n" % timeout)
        audit = load_audit(telemetry_dir)
        try:
            check_observable_clean(telemetry_dir)
            leak = ""
        except ValueError as exc:
            leak = str(exc)
            status, reason = "failed", "telemetry leak: %s" % exc
        faults = [resolve_instance_ids(fault_spec, job["cluster"], repo_root)] \
            if fault_spec is not None and suffix == "faulty" else []
        for flt in faults:
            flt["failure_class"] = MODE_TO_FAILURE_CLASS.get(flt.get("mode"), "fail_slow")
        if suffix == "faulty" and faults:
            ok, consistency = check_audit_consistency(
                {"fault_present": True, "faults": faults}, audit)
            applied = ok
            if not ok and status == "ok":
                reason = "audit inconsistency: %s" % consistency
        else:
            applied = False
            if audit:
                status, reason = "failed", "healthy run has audit records"
        entry = {
            "experiment_id": "fi_%s" % profile_name_global,
            "run_id": run_id,
            "pair_id": pair_id,
            "workload_id": workload_id,
            "workload_params": job["workload"],
            "cluster_config_id": cluster_id,
            "seed": seed,
            "git_commit": commit,
            "fault_present": suffix == "faulty" and bool(faults),
            "faults": faults,
            "failure_class": faults[0]["failure_class"] if faults else "",
            "successfully_applied": applied,
            "expected_label": expected_label_for_fault(faults[0]) if faults else "healthy",
            "status": status,
            "reason": reason + ("; %s" % leak if leak else ""),
            "total_clocks_ns": clocks,
        }
        results.append(entry)
    return results


profile_name_global = "smoke"


def main():
    global profile_name_global
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", required=True)
    parser.add_argument("--profile", default="smoke")
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout", type=int, default=1200)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--keep-inputs", action="store_true", default=False)
    args = parser.parse_args()
    profile_name_global = args.profile

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    with open(args.matrix) as f:
        matrix = yaml.safe_load(f)
    jobs = expand_profile(matrix, args.profile)
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
    for job in jobs:
        pair_id = "%s_%s_%d" % (job["workload_id"], job["cluster_id"], args.seed)
        entry = pairs.setdefault(pair_id, {"job": job, "faults": []})
        if job["fault"] is not None:
            entry["faults"].append(job["fault"])
    pending = [(pair_id, entry) for pair_id, entry in pairs.items()
               if not any((pair_id, "%s_%s" % (pair_id, suffix)) in done_pairs
                          for suffix in ["healthy"] + ["faulty%d" % i for i in range(len(entry["faults"]))])]
    print("profile %s: %d pairs (%d pending)" % (args.profile, len(pairs), len(pending)))

    manifest_file = open(manifest_path, "a")
    failures = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        future_to_job = {pool.submit(run_one, repo_root, out_root, pair_id,
                                     entry["job"], entry["faults"], args.seed,
                                     args.timeout, args.keep_inputs, commit): pair_id
                         for pair_id, entry in pending}
        for future in concurrent.futures.as_completed(future_to_job):
            try:
                entries = future.result()
            except Exception as exc:  # never lose the manifest line
                print("JOB FAILED: %s" % exc)
                failures += 1
                continue
            for entry in entries:
                manifest_file.write(json.dumps(entry) + "\n")
                manifest_file.flush()
                print("%s %s clocks=%s applied=%s" % (
                    entry["run_id"], entry["status"],
                    entry["total_clocks_ns"], entry["successfully_applied"]))
                if entry["status"] != "ok":
                    failures += 1
    manifest_file.close()
    print("done, %d failed run(s)" % failures)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
