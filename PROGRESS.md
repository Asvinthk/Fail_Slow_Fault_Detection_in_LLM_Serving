# Fault-injection extension: progress tracker

Milestone checklist (date/time UTC, test counts). Detail per command
lives in `docs/fault_injection/RUNLOG.md`.

- [x] Stage 1 inspect + baseline (2026-10-05; PD baseline 1661784496 ns x2
  identical; validate.sh PD subset 4/4 match; findings doc written; Stage 1c
  resolved: per-prefill-instance transfer path, per-pair NOT possible).
- [x] Stage 2 fault subsystem, disabled by default (unit: 15 passed
  `tests/test_fault_injection.py`; binary runs from RUNLOG: empty-faults
  byte-identical, decode-2x +39.5% reproducible, transfer +0.0% honest
  negative; semantics fixes committed after: additive last-row,
  weight-aware kv inflation, network comm scaling).
- [x] Stage 3 telemetry, additive (batch CSV full schema incl. sys,
  duration, shape snapshot, com_cycle; request CSV + ITL side file;
  rolling windows offline in ml/features.py; loader guard in manifest.py;
  unit: 3 passed `tests/test_fault_ml.py`).
- [x] Stage 4 workloads + matrix (bursty/sustained patterns + length knobs
  in workloads.py; 1P1D + 2P2D configs; matrix.yaml smoke/full; generate.py
  paired runner; manifest.py loader + audit-consistency check).
- [x] Stage 5 ML baselines (features.py observable-only + healthy residuals
  + peer ratios; run.py thresholds/residual/logistic, grouped splits,
  abstaining localization, delay + CSV/markdown report).
- [ ] Stage 6 final verification (full validate.sh green incl. 4 new fault
  scenarios; smoke matrix end-to-end; ML report on smoke data) -- needs the
  sim container (ASTRA-Sim binary); pending on this host.
