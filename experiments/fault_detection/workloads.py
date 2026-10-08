"""Tiny seeded workload generator. One file, stdlib only.

Writes JSONL rows: {input_toks, output_toks, arrival_time_ns}.
Token ids are omitted on purpose, which disables prefix-cache hits
and keeps control tight for students.

Patterns (all seeded, arrivals/lengths sampled independently of any
fault parameters):
  poisson:  gaps ~ Exp(rate_sps); rates low/medium/high set queue pressure.
  bursty:   on/off process; on-windows at high rate, off-windows silent.
  sustained: fixed small gap (near-constant stream).

Keep lengths inside the profiled sweep bounds from the profile
meta.yaml (engine_effective); beyond them the simulator extrapolates
and warns.

Usage:
  python -m experiments.fault_detection.workloads --out workloads/fi_smoke.jsonl --num 20 --seed 42
  python -m experiments.fault_detection.workloads --out w.jsonl --pattern bursty --num 60 --seed 7 --rate-sps 8 --off-s 2.0
"""

import argparse
import json
import random


def generate(num, rate_sps, seed, short_in=32, long_in=512,
             short_out=32, long_out=256, pattern="poisson",
             off_s=1.0, on_s=1.0):
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--num", type=int, default=20)
    parser.add_argument("--rate-sps", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pattern", type=str, default="poisson",
                        choices=["poisson", "bursty", "sustained"])
    parser.add_argument("--short-in", type=int, default=32)
    parser.add_argument("--long-in", type=int, default=512)
    parser.add_argument("--short-out", type=int, default=32)
    parser.add_argument("--long-out", type=int, default=256)
    parser.add_argument("--on-s", type=float, default=1.0)
    parser.add_argument("--off-s", type=float, default=1.0)
    args = parser.parse_args()
    rows = generate(args.num, args.rate_sps, args.seed,
                    short_in=args.short_in, long_in=args.long_in,
                    short_out=args.short_out, long_out=args.long_out,
                    pattern=args.pattern, off_s=args.off_s, on_s=args.on_s)
    with open(args.out, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    print("wrote %d rows to %s" % (len(rows), args.out))


if __name__ == "__main__":
    main()
