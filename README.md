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
| Held-out accuracy (random split, not session-controlled) | 100% | `results/classifier_metrics.json` |
| Cross-session accuracy, shared gesture classes (Close/Open/Side) | 72.8% | `results/cross_session.json` |
| Cross-session accuracy, all 4 classes incl. untrained Rest | 60.4% | `results/cross_session.json` |
| Shipped confidence threshold | 0.65 | `results/threshold_sweep.json` |
| Best feature representation, cross-session (wrist-relative, scaled) | 100% (+27.2pp vs. raw) | `results/feature_ablation.json` |
| Worst feature representation, cross-session (inter-landmark angles) | 66.7% (-6.2pp vs. raw) | `results/feature_ablation.json` |
| MediaPipe landmark extraction latency | 44.6ms mean / 69.4ms p95 | `results/latency.json` |
| RandomForest inference latency | 3.4ms mean / 6.0ms p95 | `results/latency.json` |
| Sustained end-to-end FPS | 19.5 | `results/latency.json` |
| False activations / minute on non-gesture footage (K=1, no debounce) | 0.287/min (3 in 10.44 min) | `results/false_activation.json` |
| Chosen debounce K, and why | K=3 (zero measured false activations; +158.7ms trigger latency) | `results/debounce_sweep.json` |

**Cross-session accuracy is real and it is much lower than the within-split
number.** `results/classifier_metrics.json`'s 100% comes from a random 80/20
shuffle that is not session-stratified — frames from the same capture burst
can land on both sides. `results/cross_session.json` trains on session 1 and
tests entirely on session 2 (different lighting, background, distance):
accuracy on the 3 gesture classes both sessions share drops to **72.8%**
(-27.2pp). Including the Rest class — which session 1 never saw, so it's
unrecognizable by construction — drops it further to 60.4%; that number
conflates "never trained on this class" with "doesn't generalize," so use the
72.8% figure as the honest cross-session estimate.

**Raw x,y coordinates are the wrong feature representation, and the data now
backs that.** `results/feature_ablation.json` compares three representations
on the same cross-session split: wrist-relative coordinates scaled by hand
bounding-box size reach **100%** cross-session accuracy (vs. 72.8% for raw
coordinates), confirming the hypothesis that translation/scale invariance
matters. Inter-landmark angles perform worse than raw (66.7%) despite also
being invariant — on this data, scale/translation normalization helped, but
discarding coordinate information for pure joint angles lost more than it
gained.

**The threshold sweep still has no signal to evaluate 0.65 against.** The
sweep runs on the same non-session-stratified split as
`classifier_metrics.json`, so precision on accepted predictions is 1.0 at
every threshold from 0.30 to 0.95 — only rejection rate moves (0% up to 0.85,
rising to 4.6% at 0.95). 0.65 is not contradicted by this data, but it isn't
validated by it either.

**MediaPipe dominates the pipeline, and the classifier is nearly free.**
Landmark extraction (44.6ms mean) is ~13x the RandomForest inference cost
(3.4ms mean), measured live over 1000 webcam frames. Sustained throughput is
19.5 FPS end-to-end.

**False activations are rare, and a small debounce eliminates them on this
footage.** Over 10.44 minutes of real non-gesturing footage (typing,
resting, moving hands naturally, no deliberate gestures), the shipped 0.65
threshold alone produced 3 false activations (0.287/min). Requiring **K=3**
consecutive agreeing predictions before firing brings that to 0 on this
footage, at a cost of +158.7ms added trigger latency (computed from the
measured per-frame pipeline cost in `results/latency.json`). This is a
single 10-minute recording, not a statistically robust estimate — treat 0.287
false activations/min and K=3 as one honest data point, not a guarantee.

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
python benchmarks/cross_session_eval.py      # writes results/cross_session.json
python benchmarks/feature_ablation.py        # writes results/feature_ablation.json
python benchmarks/false_activation.py <video>  # writes results/false_activation.json, needs cv2/mediapipe
python benchmarks/debounce_sweep.py <video>    # writes results/debounce_sweep.json, needs cv2/mediapipe
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
- **Two capture sessions, one lightly documented.** `data/sessions/session_1`
  has no recorded lighting/background/distance manifest (it predates the
  manifest system). `session_2` does have a manifest ("lamp only", "plain
  white wall", "90cm") but its own notes field says "TESTING" and it was
  captured in about 76 seconds — treat it as a real second condition (the
  cross-session accuracy drop is consistent with that), but not as a
  carefully controlled second session on the level a rigorous ablation would
  want. A third, more deliberate session would strengthen this.
- **3 trained gesture classes plus Rest.** The model now includes a 4th
  "Rest" (no-gesture) class, captured only in session 2 (123 usable samples
  after landmark detection). Because session 1 never saw Rest,
  `results/cross_session.json`'s all-4-class number can't be compared
  apples-to-apples with the within-split number — see `RESULTS.md`.
- **False-activation rate is measured on a single 10.44-minute recording.**
  0.287/min at K=1, 0 at K=3 — real, but one data point, not a distribution.
  A second, independently-recorded session of non-gesture footage would make
  this more trustworthy.
- **`static_image_mode=True`** is used for every frame in the live inference
  path, forcing full MediaPipe detection instead of its cheaper tracking
  mode. This is a likely contributor to the 44.6ms mean extraction latency
  measured in `results/latency.json`; not changed here since that would be a
  behavior change beyond measurement.
