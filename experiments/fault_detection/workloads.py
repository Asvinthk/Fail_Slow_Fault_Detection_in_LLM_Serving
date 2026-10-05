"""Tiny seeded workload generator. One file, no dependencies.

Writes JSONL rows: {input_toks, output_toks, arrival_time_ns}.
Token ids are omitted on purpose, which disables prefix-cache hits
and keeps control tight for students.

Usage:
  python -m experiments.fault_detection.workloads --out workloads/fi_smoke.jsonl --num 20 --seed 42
"""

import argparse
import json
import random


def generate(num, rate_sps, seed, short_in=32, long_in=512, short_out=32, long_out=256):
    rnd = random.Random(seed)
    time_ns = 0
    rows = []
    for _ in range(num):
        gap_s = rnd.expovariate(rate_sps) if rate_sps > 0 else 0.0
        time_ns += int(gap_s * 1e9)
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
    args = parser.parse_args()
    rows = generate(args.num, args.rate_sps, args.seed)
    with open(args.out, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    print("wrote %d rows to %s" % (len(rows), args.out))


if __name__ == "__main__":
    main()
