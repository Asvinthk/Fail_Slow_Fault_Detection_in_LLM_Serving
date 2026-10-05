# Run log (commands + outputs, failures included)

## Stage 1 — inspect and baseline

- `git log -1 --oneline` -> `a4053bc Lead the README with a Latest News section`, branch `main`, submodule `astra-sim d346994`.
- `python -m serving --help` -> flags verified (no `--fault-config` upstream; added by this work).
- `Test-Path astra-sim/build/.../AnalyticalAstra` -> `False`. NOT in sim container (Windows host). ASTRA-Sim binary missing.
- `serving/validate.sh` and PD baseline runs need the binary, so they could not run here. Recorded as NOT RUN (binary missing), not as success.
- `python -m pytest tests/test_fault_injection.py -v` -> exit 0, 7 passed (pure unit tests, no binary needed).
