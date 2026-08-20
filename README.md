# Hand Gesture Browser Opener

A hand-gesture recognition pipeline: a webcam feed is processed with MediaPipe
Hands to extract 21 landmark positions per detected hand, a scikit-learn
RandomForestClassifier classifies the resulting feature vector into one of a
small set of gestures, and the live inference script opens one of three
hardcoded URLs when two different gestures are detected in sequence above a
confidence threshold. This is a measured demo, not a product — see
[LIMITATIONS](#limitations) below and [RESULTS.md](RESULTS.md) for the full
set of measured numbers and how they were produced.

## Results

Full detail and provenance for every number below is in
[RESULTS.md](RESULTS.md). Summary:

| Metric | Value | Source |
|---|---|---|
| Held-out classification accuracy (within-session split) | 100% | `results/classifier_metrics.json` |
| Cross-session accuracy | not yet measured (needs a second capture session) | `results/cross_session.json` (blocked) |
| Shipped confidence threshold | 0.65 | `results/threshold_sweep.json` |
| MediaPipe landmark extraction latency | 44.6ms mean / 69.4ms p95 | `results/latency.json` |
| RandomForest inference latency | 3.4ms mean / 6.0ms p95 | `results/latency.json` |
| Sustained end-to-end FPS | 19.5 | `results/latency.json` |
| False activations / minute on non-gesture footage | not yet measured (needs recorded footage + Rest class) | `results/false_activation.json` (not present) |

**The within-session accuracy is inflated and known to be inflated.** The
current split is drawn from a single capture session (`data/sessions/session_1`),
so training and test data share lighting, background, camera distance, and
the same hand. 100% accuracy on that split says the classifier memorized this
session's conditions, not that it generalizes. A corrected, cross-session
number is pending — see the Limitations section.

**The threshold sweep did not validate 0.65 as a good operating point.**
Because the sweep runs on the same leaked within-session split, precision on
accepted predictions is 1.0 at every threshold from 0.30 to 0.95 — the sweep
has no signal to distinguish thresholds by accuracy, only by rejection rate
(0% up to 0.70, rising to 6.1% at 0.95). 0.65 is not contradicted by this
data, but it isn't backed by it either; a real threshold decision needs the
cross-session split.

**MediaPipe dominates the pipeline, and the classifier is nearly free.**
Landmark extraction (44.6ms mean) is ~13x the RandomForest inference cost
(3.4ms mean), measured live over 1000 webcam frames. Sustained throughput is
19.5 FPS end-to-end.

## Tech Stack

- Python (3.13 for training/analysis, 3.11 for capture/live inference — see
  [requirements.txt](requirements.txt))
- MediaPipe Hands — landmark extraction
- OpenCV — video capture
- scikit-learn — RandomForestClassifier
- Python `webbrowser` stdlib — opens the mapped URL

## Reproduction

```
pip install -r requirements.txt
python train_data.py                        # trains on data.pickle, writes results/classifier_metrics.json
python benchmarks/threshold_sweep.py         # writes results/threshold_sweep.json and .png
python benchmarks/latency_benchmark.py       # live webcam benchmark, needs cv2/mediapipe
```

`data.pickle` is not committed (see `.gitignore`) — regenerate it from raw
captures with `processing_data.py`, or capture your own with
`collect_session.py`.

## Demo

_GIF placeholder — record a short clip of a gesture triggering a browser
open and drop it here._

## Limitations

- **Single user.** All data was captured from one person's hand; the
  classifier has never seen anyone else's hand shape or gesture style.
- **One capture session for the current results.** `data/sessions/session_1`
  is the only session with a trained-on model; a second session under
  different conditions is planned (see `benchmarks/cross_session_eval.py`,
  currently blocked) but not yet recorded.
- **Controlled lighting and background.** Session 1 has no recorded
  lighting/background/distance manifest (it predates the manifest system);
  informally, it was captured in one sitting under one lighting setup.
- **3 gesture classes plus a not-yet-trained 4th.** The shipped model
  recognizes "Close" (fist), "Open" (palm), and "Side" only. A 4th "Rest"
  (no-gesture) class has been added to the capture tooling
  (`collect_session.py`) but not yet captured or trained, so the current
  model has no notion of "doing nothing" — every frame with a detected hand
  is forced into one of the 3 gesture classes.
- **`static_image_mode=True`** is used for every frame in the live inference
  path, forcing full MediaPipe detection instead of its cheaper tracking
  mode. This is a likely contributor to the 44.6ms mean extraction latency
  measured in `results/latency.json`; not changed here since that would be a
  behavior change beyond measurement.
