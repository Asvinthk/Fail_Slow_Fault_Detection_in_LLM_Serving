# Fault-detection experiments

Research extension on top of LLMServingSim 2.0. The project **extends**
the simulator (fault hooks, telemetry, experiment harness) rather than
replacing its scheduler or network backend, which are upstream code
and stay byte-identical when no fault option is given.

## Area map

- `workloads.py` — seeded Poisson/bursty/sustained JSONL generator.
- `matrix.yaml` — experiment matrix (`smoke` and `full` profiles).
- `generate.py` — paired healthy/faulty runner; writes `manifest.jsonl`.
- `manifest.py` — manifest loader, observable-leak guard, audit check.
- `ml/features.py` — observable-only window features, healthy-only
  residuals, peer ratios.
- `ml/run.py` — fixed-threshold, residual-threshold, and logistic
  baselines with abstaining localisation stub; one-command report.
- `smoke/` — frozen reproducibility bundle for the smoke evaluation.
- `example_faults/` — small committed fault configs for manual runs.

## Contribution vs upstream

| Area | Upstream | Our contribution |
|---|---|---|
| Fault injection | none | `serving/core/fault_injection.py`: validator, severity profiles, targeting, trace-row scaling, audit entries |
| Fault-aware routing / stall handling | plain routing + scheduling | `router.py`: stall skip, hold-and-retry of decode transfers; `__main__.py`: schedule guard, held queues, fault wake-ups |
| Telemetry | per-request CSV only | `--telemetry-output` batch/request CSVs + ITL side file; `com_cycle` in `controller.parse_output`; schedule-time batch snapshots |
| Experiment generation | workload generators only | `workloads.py` patterns/seeds, `matrix.yaml`, `generate.py` paired runner |
| Paired healthy/faulty runs | none | shared workload file + `pair_id` per (workload, cluster, seed) |
| Manifest and leak guard | none | `manifest.py`: schema, loader, observable-column guard, audit-consistency check |
| ML features | none | `ml/features.py`: windows, healthy-only least-squares residuals, peer ratios, ground-truth column ban |
| Detection baselines | none | `ml/run.py`: fixed threshold, residual threshold, logistic regression, abstaining localisation stub, CSV+markdown report |
| Tests | no suite | `tests/test_fault_injection.py`, `tests/test_fault_ml.py` |
| Documentation | site + README | `docs/fault_injection/` (7 files), `smoke/` bundle, this file |

## Scratch / helper files (not research contributions)

- `check_audit.py` — 9-line audit inspector used during debugging.
- `fix_lines.py` — 9-line CRLF-to-LF converter for container shell scripts.

## Honest scope notes

- No MLP, gradient boosting, macro-F1/confusion-matrix, top-1/top-3,
  per-family-FPR, or mechanism-classification model is implemented.
- Localisation is a stub that echoes ground truth when confident.
- The detector uses simulator-visible serving telemetry only; physical
  hardware counters (GPU power/thermal, SM/GPU utilisation, NIC/RDMA,
  clocks/throttling) are not exposed through the simulator path.
- "Mechanism" means simulator injection mode, never physical root cause.
