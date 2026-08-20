# Results

Every number below was produced by running a script committed in this repo
against real captured data on the benchmark machine (Intel Core Ultra 7 256V,
8 logical CPUs, 16 GB RAM, Windows 11, CPU only). No number here was
estimated or invented. Config (seed, library versions, sample counts) is
recorded inside each results file, not repeated in full here.

## Measured

| Metric | Value | Produced by | Results file |
|---|---|---|---|
| Held-out accuracy, random 80/20 split (not session-stratified) | 100% (4 classes, 1623 samples: 500/500/500/123) | `train_data.py` | `results/classifier_metrics.json` |
| Within-session accuracy (session_1 only, stratified 80/20) | 100% (3 classes, 900 samples) | `benchmarks/cross_session_eval.py` | `results/cross_session.json` |
| Cross-session accuracy, shared classes only (Close/Open/Side) | 72.8% | `benchmarks/cross_session_eval.py` | `results/cross_session.json` |
| Cross-session accuracy, all 4 test classes (incl. untrained Rest) | 60.4% | `benchmarks/cross_session_eval.py` | `results/cross_session.json` |
| Cross-session accuracy delta, shared classes (within - cross) | -27.2pp | `benchmarks/cross_session_eval.py` | `results/cross_session.json` |
| Feature ablation: raw x,y coordinates (baseline) | 72.8% cross-session, dim=42 | `benchmarks/feature_ablation.py` | `results/feature_ablation.json` |
| Feature ablation: wrist-relative, bbox-scaled | 100.0% cross-session, dim=42 (+27.2pp) | `benchmarks/feature_ablation.py` | `results/feature_ablation.json` |
| Feature ablation: inter-landmark angles | 66.7% cross-session, dim=20 (-6.2pp) | `benchmarks/feature_ablation.py` | `results/feature_ablation.json` |
| Threshold sweep: precision on accepted predictions | 1.0 at every threshold 0.30-0.95 | `benchmarks/threshold_sweep.py` | `results/threshold_sweep.json`, `.png` |
| Threshold sweep: rejection rate at shipped threshold (0.65) | 0.0% | `benchmarks/threshold_sweep.py` | `results/threshold_sweep.json` |
| Threshold sweep: rejection rate at 0.95 | 4.6% (15/325 frames) | `benchmarks/threshold_sweep.py` | `results/threshold_sweep.json` |
| Frame capture latency | mean 4.9ms, p50 2.6ms, p95 18.5ms | `benchmarks/latency_benchmark.py` | `results/latency.json` |
| MediaPipe landmark extraction latency | mean 44.6ms, p50 42.1ms, p95 69.4ms | `benchmarks/latency_benchmark.py` | `results/latency.json` |
| Feature vector construction latency | mean 0.025ms, p50 0.020ms, p95 0.043ms | `benchmarks/latency_benchmark.py` | `results/latency.json` |
| RandomForest predict_proba latency | mean 3.4ms, p50 2.9ms, p95 6.0ms | `benchmarks/latency_benchmark.py` | `results/latency.json` |
| Sustained end-to-end FPS | 19.52 (1000 frames, 51.2s wall time) | `benchmarks/latency_benchmark.py` | `results/latency.json` |
| MediaPipe-to-classifier latency ratio | ~13x (44.6ms / 3.4ms) | derived from `results/latency.json` | `results/latency.json` |
| False activations, K=1 (no debounce), on 10.44 min of non-gesture footage | 3 frames = 0.287/min, threshold 0.65 | `benchmarks/false_activation.py` | `results/false_activation.json` |
| False activations, K=2 | 1 event = 0.096/min, +105.8ms trigger latency | `benchmarks/debounce_sweep.py` | `results/debounce_sweep.json` |
| False activations, K=3 (chosen operating point) | 0 events, +158.7ms trigger latency | `benchmarks/debounce_sweep.py` | `results/debounce_sweep.json` |
| False activations, K=4-8 | 0 events, +211.6ms to +423.3ms trigger latency | `benchmarks/debounce_sweep.py` | `results/debounce_sweep.json` |

### How to read the accuracy numbers together

- **100% (classifier_metrics.json)** is not a real generalization estimate.
  It's a random shuffle across whatever sessions exist in `data.pickle` at
  run time, not stratified by session, so frames from the same capture burst
  can land on both sides of the split.
- **100% within-session (cross_session.json)** is the same style of number,
  restricted to session_1 alone (stratified 80/20) — still leaked in the same
  way, kept for direct comparison against the cross-session numbers below it.
- **72.8% cross-session, shared classes** is the first non-leaked estimate in
  this project: trained entirely on session_1, evaluated entirely on
  session_2, restricted to the 3 gesture classes both sessions have. This is
  the number to cite if you need one honest accuracy figure.
- **60.4% cross-session, all 4 classes** additionally includes the Rest
  class, which session_1 never saw during training — the model can't
  possibly predict a class it was never trained on, so this number blends
  real generalization failure with an unwinnable class. Not a fair "how good
  is the model" number; included for completeness only.
- **Feature ablation numbers** all use the shared-classes-only basis above,
  so they're comparable to the 72.8% baseline.

### Data quality caveat on session_2

`data/sessions/session_2/meta.json` records itself with `"notes": "TESTING"`
and a ~76-second capture window for all 800 images across 4 classes. It does
have a real, different manifest from session_1 (lamp-only lighting, plain
white wall, 90cm distance vs. session_1's undocumented conditions), and the
cross-session accuracy drop is large and directionally consistent with a real
capture-condition shift — so these numbers are reported as genuine. But
session_2 was not a deliberately slow, careful capture, and a third session
done more rigorously would be a reasonable next step before treating 72.8%
as a stable estimate rather than a single data point.

### Data quality caveat on the false-activation footage

`false_activation.py` and `debounce_sweep.py` ran against one 10.44-minute
recording (18,763 frames at ~30fps) of natural non-gesturing activity. 3
false activations at K=1 out of 18,763 frames is a small-sample result — a
single unlucky or lucky 10-minute window could plausibly shift the rate.
Treat 0.287/min and the choice of K=3 as one honest measurement, not a
statistically robust rate. A second, independently-recorded footage session
would strengthen this.

## Not measured — nothing left that's blocked

All harnesses described in the original task list have now been run against
real data. The only thing not done is optional strengthening (see below),
not a missing measurement.

## Optional next steps to strengthen existing numbers

None of these are blockers — everything in the original task list is
measured. These would tighten confidence intervals on numbers that are
currently single data points:

1. A third capture session, done deliberately (not the ~76-second
   "TESTING" pass that session_2 was), to check whether 72.8% cross-session
   accuracy and the feature-ablation deltas hold up on a second independent
   session pair.
2. A second, independently-recorded non-gesture footage session, to check
   whether 0.287 false-activations/min and the K=3 operating point hold up
   on different natural behavior/lighting.
