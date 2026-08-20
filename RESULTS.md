# Results

Every number below was produced by running a script committed in this repo
against real captured data on the benchmark machine (Intel Core Ultra 7 256V,
8 logical CPUs, 16 GB RAM, Windows 11, CPU only). No number here was
estimated or invented. Config (seed, library versions, sample counts) is
recorded inside each results file, not repeated in full here.

## Measured

| Metric | Value | Produced by | Results file |
|---|---|---|---|
| Held-out classification accuracy | 100% (60/60 per class, 3 classes) | `train_data.py` | `results/classifier_metrics.json` |
| Per-class precision/recall/F1 | 1.0 / 1.0 / 1.0 for all 3 classes | `train_data.py` | `results/classifier_metrics.json` |
| Confusion matrix | perfect diagonal, 60/60/60 | `train_data.py` | `results/classifier_metrics.json` |
| Threshold sweep: precision on accepted predictions | 1.0 at every threshold 0.30-0.95 | `benchmarks/threshold_sweep.py` | `results/threshold_sweep.json`, `.png` |
| Threshold sweep: rejection rate at shipped threshold (0.65) | 0.0% | `benchmarks/threshold_sweep.py` | `results/threshold_sweep.json` |
| Threshold sweep: rejection rate at 0.95 | 6.1% (11/180 frames) | `benchmarks/threshold_sweep.py` | `results/threshold_sweep.json` |
| Frame capture latency | mean 4.9ms, p50 2.6ms, p95 18.5ms | `benchmarks/latency_benchmark.py` | `results/latency.json` |
| MediaPipe landmark extraction latency | mean 44.6ms, p50 42.1ms, p95 69.4ms | `benchmarks/latency_benchmark.py` | `results/latency.json` |
| Feature vector construction latency | mean 0.025ms, p50 0.020ms, p95 0.043ms | `benchmarks/latency_benchmark.py` | `results/latency.json` |
| RandomForest predict_proba latency | mean 3.4ms, p50 2.9ms, p95 6.0ms | `benchmarks/latency_benchmark.py` | `results/latency.json` |
| Sustained end-to-end FPS | 19.52 (1000 frames, 51.2s wall time) | `benchmarks/latency_benchmark.py` | `results/latency.json` |
| MediaPipe-to-classifier latency ratio | ~13x (44.6ms / 3.4ms) | derived from `results/latency.json` | `results/latency.json` |

**Important caveat on the first four rows:** the accuracy/precision/recall/
confusion-matrix numbers and the threshold sweep are all computed on a
single-session split (`data/sessions/session_1`), which means train and test
share lighting, background, camera distance, and the same hand. The 100%
accuracy and the flat 1.0 precision across every threshold are consequences
of that leakage, not evidence of a well-tuned classifier or a validated
threshold. Do not read the threshold sweep as endorsing 0.65 — it has no
signal at any threshold on this data. See below for the corrected,
cross-session numbers, which cannot be measured yet.

## Not measured — blocked on data collection

These harnesses are built, committed, and run — each one correctly reports a
`"status": "blocked"` (or exits with an explanatory error) instead of
producing a fabricated number.

| Metric | Blocked because | Harness | Results file |
|---|---|---|---|
| Cross-session accuracy (within-session vs. cross-session) | Only one capture session (`session_1`) exists. Needs a second session recorded under deliberately different lighting/background/distance, on a different day, via `collect_session.py`. | `benchmarks/cross_session_eval.py` | `results/cross_session.json` (status: blocked) |
| Feature representation ablation (raw vs. wrist-relative-scaled vs. inter-landmark-angle, cross-session accuracy) | Depends on the same second session as above. | `benchmarks/feature_ablation.py` | `results/feature_ablation.json` (status: blocked) |
| False activations per minute on non-gesture footage | No Rest-class training data exists yet (model.p only has classes Close/Open/Side), and no natural non-gesturing footage has been recorded. Needs both: capture a Rest class via `collect_session.py`, retrain, then record ≥10 minutes of real footage (typing, drinking, talking with hand movement). | `benchmarks/false_activation.py` | not present — script exits with a clear error rather than running against an untrained model |
| Debounce sweep (false-activation rate vs. added latency across K=1-8) | Same two blockers as above, plus depends on `false_activation.py`'s footage. | `benchmarks/debounce_sweep.py` | not present — same guard |

## What's needed to unblock everything

1. Record session 2 with `collect_session.py`: different lighting, different
   background, different distance, different day. This also captures the
   new Rest (no-gesture) class in the same pass.
2. Retrain (`train_data.py`) once the Rest class exists.
3. Record ≥10 minutes of real, natural non-gesturing footage (a video file)
   for the false-activation and debounce benchmarks.
4. Re-run `processing_data.py` → `benchmarks/cross_session_eval.py` →
   `benchmarks/feature_ablation.py` → `benchmarks/false_activation.py` →
   `benchmarks/debounce_sweep.py`, in that order.
