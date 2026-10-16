"""Reproducible full-day arrival generator (NHPP with diurnal shape).

Writes JSONL rows in the existing workload schema:
  {input_toks, output_toks, arrival_time_ns}

Arrival model: non-homogeneous Poisson process via thinning.
  rate(t) = base_sps * (1 + morning(t) + noon_peak(t) + afternoon(t)
                        + evening(t)) + bursts
  - morning:   linear ramp from day_start to noon - peak_width
  - noon_peak: Gaussian centred at peak_hour (default 12.0), height
    peak_height, width peak_sigma_hours. Configurable; set peak_height=0
    to disable.
  - afternoon: exponential decay after the peak.
  - evening:   low flat level (fraction of base).
  - bursts:    optional short seeded bursts (count, rate multiplier,
    duration).

All randomness from random.Random(seed): reproducible. Identical config +
seed gives byte-identical output.

End-of-day policy is NOT enforced here: every generated request carries an
arrival within [0, horizon]. Draining vs truncating leftover queues is a
simulator-runner concern (see --grace-s and RUNLOG conventions). The
generator records horizon_s in a sidecar manifest line when --manifest is
given so runners can size the grace period.

Simulated time vs wall-clock: arrival_time_ns is simulated nanoseconds.
Generating a 24h horizon is cheap (thinning loop); SIMULATING it costs
ASTRA-Sim handshakes per batch -- measure with short horizons first.

Usage:
  python -m experiments.fault_detection.day_workload --out w.jsonl --seed 7
  python -m experiments.fault_detection.day_workload --out w.jsonl --horizon-h 24 --base-sps 2.0 --peak-height 3.0 --peak-hour 12.0 --seed 7 --manifest w.manifest.json
"""

import argparse
import json
import math
import random


def rate_at(t_s, cfg):
    """Instantaneous arrival rate (requests/s) at sim second t_s."""
    base = cfg["base_sps"]
    horizon = cfg["horizon_s"]
    peak_h = cfg["peak_hour"] * 3600.0
    sigma = max(cfg["peak_sigma_h"] * 3600.0, 1.0)
    # Morning ramp: 0 -> 1 over [0.5h, peak - sigma].
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
    rate = base * (morning * (1.0 + noon) * aft) * evening
    return max(rate, 0.0)


def generate(cfg):
    rnd = random.Random(cfg["seed"])
    horizon_s = cfg["horizon_s"]
    # Thinning: candidate events at max_rate, accept with p = rate/max.
    peak_max = cfg["base_sps"] * (1.0 + cfg["peak_height"]) + 1e-9
    burst_max = 0.0
    for b in cfg.get("bursts", []):
        burst_max = max(burst_max, cfg["base_sps"] * b.get("rate_mult", 2.0))
    max_rate = max(peak_max, burst_max, 1e-9)
    # Burst windows (seeded placement unless explicit start given).
    bursts = []
    for b in cfg.get("bursts", []):
        start = b.get("start_s")
        if start is None:
            start = rnd.uniform(0, max(horizon_s - b.get("dur_s", 60.0), 0.0))
        bursts.append({"start_s": start, "dur_s": b.get("dur_s", 60.0),
                       "rate_mult": b.get("rate_mult", 2.0)})
    t = 0.0
    arrivals = []
    while True:
        t += rnd.expovariate(max_rate)
        if t >= horizon_s:
            break
        r = rate_at(t, cfg)
        for b in bursts:
            if b["start_s"] <= t < b["start_s"] + b["dur_s"]:
                r += cfg["base_sps"] * (b["rate_mult"] - 1.0)
        if rnd.random() < r / max_rate:
            arrivals.append(t)
    rows = []
    for t in arrivals:
        rows.append({
            "input_toks": rnd.randint(cfg["short_in"], cfg["long_in"]),
            "output_toks": rnd.randint(cfg["short_out"], cfg["long_out"]),
            "arrival_time_ns": int(t * 1e9),
        })
    return rows, {"num_requests": len(rows), "horizon_s": horizon_s,
                  "bursts": bursts, "seed": cfg["seed"]}


def default_config(args):
    bursts = []
    if args.bursts > 0:
        for _ in range(args.bursts):
            bursts.append({"dur_s": args.burst_dur_s,
                           "rate_mult": args.burst_rate_mult})
    return {
        "horizon_s": args.horizon_h * 3600.0,
        "base_sps": args.base_sps,
        "peak_hour": args.peak_hour,
        "peak_height": args.peak_height,
        "peak_sigma_h": args.peak_sigma_h,
        "afternoon_tau_h": args.afternoon_tau_h,
        "evening_frac": args.evening_frac,
        "bursts": bursts,
        "seed": args.seed,
        "short_in": args.short_in,
        "long_in": args.long_in,
        "short_out": args.short_out,
        "long_out": args.long_out,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--horizon-h", type=float, default=24.0)
    p.add_argument("--base-sps", type=float, default=2.0)
    p.add_argument("--peak-hour", type=float, default=12.0)
    p.add_argument("--peak-height", type=float, default=3.0,
                   help="0 disables the noon peak")
    p.add_argument("--peak-sigma-h", type=float, default=1.5)
    p.add_argument("--afternoon-tau-h", type=float, default=3.0)
    p.add_argument("--evening-frac", type=float, default=0.4)
    p.add_argument("--bursts", type=int, default=0)
    p.add_argument("--burst-dur-s", type=float, default=60.0)
    p.add_argument("--burst-rate-mult", type=float, default=2.0)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--short-in", type=int, default=32)
    p.add_argument("--long-in", type=int, default=512)
    p.add_argument("--short-out", type=int, default=32)
    p.add_argument("--long-out", type=int, default=256)
    p.add_argument("--manifest", type=str, default=None)
    args = p.parse_args()
    cfg = default_config(args)
    rows, info = generate(cfg)
    with open(args.out, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    info.update({k: cfg[k] for k in ("base_sps", "peak_hour",
                                    "peak_height", "peak_sigma_h",
                                    "afternoon_tau_h", "evening_frac")})
    if args.manifest:
        with open(args.manifest, "w") as f:
            json.dump(info, f, indent=2)
    print("wrote %d rows (%.2fh horizon) to %s" % (
        len(rows), args.horizon_h, args.out))


if __name__ == "__main__":
    main()
