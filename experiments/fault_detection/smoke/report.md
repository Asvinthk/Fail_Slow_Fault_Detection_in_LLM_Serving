# Smoke evaluation report (verified from frozen data)

Source: `data/` (5 runs) + `manifest.jsonl`, produced by the command in
`README.md` (seed 42). Metrics recomputed independently from the
telemetry (see verification note) and match committed `metrics.csv`.
40 windows total: **22 healthy, 18 anomalous** (2 s windows).

## Verified counts (whole smoke set)

| Detector | TP | FP | FN | TN | P | R | F1 | FPR (denominator) |
|---|---|---|---|---|---|---|---|---|
| Fixed threshold | 6 | 0 | 12 | 22 | 1.000 | 0.333 | 0.500 | 0.000 (0/22 healthy) |
| Residual-only | 7 | 1 | 11 | 21 | 0.875 | 0.389 | 0.538 | 0.045 (1/22 healthy) |

Train/test split used in `metrics.csv` (pair-grouped): test = 9
anomalous + 8 healthy windows; train = 9 + 14. Cross-checked on all
six rows, e.g. residual test FPR 0.125 = 1 false alarm in 8 healthy
test windows; logistic test FPR 0.625 = 5/8. Fixed threshold: test
P 1.000 / R 0.556 / FPR 0.000 — higher precision and zero false
alarms, but it misses the subtle handoff fault. Logistic regression
test P 0.286 / R 0.222 — preliminary negative result; insufficient
data (40 windows) is a possible explanation, overfitting is not
proven. Localisation 0.475 / abstention 0.075 / selective 0.594;
mean detection delay 1.0 s against the 2 s window floor.

## Headline: healthy-heavy vs faulty-light

Heavy-but-healthy decode windows (24-req Poisson run) sit at
11–17 ms duration with residuals near zero (−0.86…+0.12 ms);
faulty-light decode windows (10-req run, decode ×2.0) sit at
~22 ms with residuals ≈ +11 ms and peer ratios 1.8–2.3:

| Condition | Workload | Fault | Window | Duration | Residual | Peer ratio | Fixed | Residual |
|---|---|---|---|---|---|---|---|---|
| healthy-heavy | heavy_poisson | none | 2–6 s | 11.4–16.7 ms | −0.9…+0.1 ms | 0.7–1.4 | quiet | quiet |
| faulty-light | light_poisson | decode ×2.0 | 2–10 s | ~22 ms | ≈ +11 ms | 1.8–2.3 | alarm | alarm |

Caveat that caps every recall number: windows on the *prefill*
instance of a faulty run are labeled anomalous by run but carry no
mechanism effect (the fault targets decode), so no duration- or
residual-based detector can catch them — e.g. faulty-light prefill
windows read ~10–12 ms with residuals ≈ 0. Maximum achievable recall
on this split is therefore bounded by label construction, not by
detector quality.

## Interpretation (do not overstate)

The smoke sample is consistent with the hypothesis that
workload-normalised residuals help distinguish faults from workload
variation (residual test F1 0.750 vs fixed 0.714, with the miss
patterns differing), but n=40 windows is too small to establish
superiority. The fixed threshold's perfect precision / zero FPR is
reported alongside, not hidden. No "residuals are better" general
conclusion is made.

## What is NOT in this bundle

No MLP, gradient-boosted, or other learned detector; no genuine
localisation model (the 0.475 figure comes from a stub that echoes
ground truth when confident — see `ml/run.py` — and is not a model
result); no macro-F1, confusion matrix, top-1/top-3, per-family FPR,
or mechanism-classification evaluation; no serving-overhead
measurement; no full-matrix or Qwen runs. Those are pending work.
