"""Tiny audit checker: which instances did a fault hit?"""
import json
import sys

path = sys.argv[1]
rows = [json.loads(line) for line in open(path) if line.strip()]
print("records:", len(rows))
print("instances:", sorted(set(r.get("instance") for r in rows)))
print("modes:", sorted(set(r.get("mode") for r in rows)))
