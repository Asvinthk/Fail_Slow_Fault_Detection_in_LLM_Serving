"""Find the arrival rate that puts the decode pool at a target utilisation.

  python -m experiments.fault_detection.calibrate \
      --cluster configs/cluster/pd_2p2d_small.json --target-u 0.75 \
      --out experiments/out/pilot_2p2d --num-reqs 40

Why (review Q1/Q3): the experiment unit must be a LOAD-MATCHED window,
not "a day". A 24 h trace at the documented 2 req/s is ~173k requests,
and measured cost is ~127 s wall per request, so a full-volume day is
~250 days of wall clock. The defensible alternative is to choose lambda
so the decode pool sits at a stated utilisation, run a few simulated
minutes, and say so explicitly.

Method: bisection on lambda. Each probe generates a seeded Poisson trace,
runs the simulator with telemetry, and measures utilisation from the
batch durations (regime.py). The search stops after --iters probes or
when the measured U is within --tol of the target, and the probe closest
to the target is reported.

The output JSON is the regime record that belongs in the manifest:
lambda, U_prefill, U_decode, TPOT p50, span, request count.
"""

import argparse
import json
import os
import subprocess
import sys

from experiments.fault_detection.regime import regime_from_telemetry
from experiments.fault_detection.workloads import generate as generate_workload


def run_probe(repo_root, out_root, cluster, lam, args, tag):
    probe_dir = os.path.join(out_root, tag)
    os.makedirs(probe_dir, exist_ok=True)
    workload_path = os.path.join(probe_dir, "workload.jsonl")
    rows = generate_workload(num=args.num_reqs, rate_sps=lam, seed=args.seed,
                             short_in=args.short_in, long_in=args.long_in,
                             short_out=args.short_out, long_out=args.long_out)
    with open(workload_path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    csv_path = os.path.join(probe_dir, "sim.csv")
    telemetry_dir = os.path.join(probe_dir, "telemetry")
    cmd = [sys.executable, "-m", "serving",
           "--cluster-config", cluster,
           "--dtype", "bfloat16", "--block-size", "16",
           "--dataset", os.path.relpath(workload_path, repo_root),
           "--output", os.path.relpath(csv_path, repo_root),
           "--num-reqs", str(len(rows)),
           "--run-id", tag,
           "--telemetry-output", os.path.relpath(telemetry_dir, repo_root)]
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=repo_root,
                          timeout=args.timeout)
    if proc.returncode != 0:
        raise RuntimeError("probe %s failed:\n%s" % (tag, (proc.stdout + proc.stderr)[-2000:]))
    regime = regime_from_telemetry(telemetry_dir)
    regime.update({"lambda_requested": lam, "tag": tag})
    return regime


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cluster", required=True)
    p.add_argument("--target-u", type=float, default=0.75)
    p.add_argument("--out", required=True)
    p.add_argument("--num-reqs", type=int, default=40)
    p.add_argument("--iters", type=int, default=5)
    p.add_argument("--tol", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=11)
    p.add_argument("--timeout", type=int, default=1800)
    p.add_argument("--short-in", type=int, default=32)
    p.add_argument("--long-in", type=int, default=512)
    p.add_argument("--short-out", type=int, default=32)
    p.add_argument("--long-out", type=int, default=256)
    args = p.parse_args()

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    out_root = args.out if os.path.isabs(args.out) else os.path.join(repo_root, args.out)
    os.makedirs(out_root, exist_ok=True)

    lo, hi = 0.05, 64.0
    probes = []
    for i in range(args.iters):
        lam = (lo * hi) ** 0.5  # geometric midpoint: rates span decades
        regime = run_probe(repo_root, out_root, args.cluster, lam, args,
                           "probe_%02d" % i)
        probes.append(regime)
        print("probe %d: lambda=%.3f/s -> U_decode=%.3f U_prefill=%.3f "
              "TPOT=%.2fms span=%.1fs"
              % (i, lam, regime["u_decode"], regime["u_prefill"],
                 regime["tpot_p50_ms"], regime["span_s"]))
        if abs(regime["u_decode"] - args.target_u) <= args.tol:
            break
        if regime["u_decode"] < args.target_u:
            lo = lam
        else:
            hi = lam

    best = min(probes, key=lambda r: abs(r["u_decode"] - args.target_u))
    record = {
        "cluster": args.cluster,
        "target_u": args.target_u,
        "chosen": best,
        "probes": probes,
        "note": ("Load-matched pilot: lambda chosen for the decode-pool "
                 "utilisation above. Report this regime with every detection "
                 "number measured on this cluster."),
    }
    path = os.path.join(out_root, "regime.json")
    with open(path, "w") as f:
        json.dump(record, f, indent=2)
    print("chosen lambda=%.3f/s (U_decode=%.3f, target %.2f) -> %s"
          % (best["lambda_requested"], best["u_decode"], args.target_u, path))
    return 0


if __name__ == "__main__":
    sys.exit(main())
