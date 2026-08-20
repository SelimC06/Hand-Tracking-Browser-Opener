import argparse
import json
import os
import platform

import sklearn

from _footage_utils import ROOT, get_or_build_trace, load_model, require_rest_class

THRESHOLD = 0.65
MIN_FOOTAGE_SECONDS = 10 * 60


def main():
    parser = argparse.ArgumentParser(description="Measure false activations per minute on natural non-gesturing footage.")
    parser.add_argument("video_path", help="Path to a video file of natural non-gesturing hand movement "
                                            "(typing, drinking, talking with hand movement), at least 10 minutes long.")
    args = parser.parse_args()

    model = load_model()
    require_rest_class(model)

    trace, fps = get_or_build_trace(args.video_path, model, threshold=THRESHOLD)

    duration_s = trace[-1]["timestamp_s"] if trace else 0.0
    if duration_s < MIN_FOOTAGE_SECONDS:
        raise SystemExit(
            f"Footage is only {duration_s:.1f}s long; the task requires at least "
            f"{MIN_FOOTAGE_SECONDS}s (10 minutes) of natural non-gesturing footage. "
            "Record more footage before running this benchmark."
        )

    false_activation_frames = [f for f in trace if f["is_gesture_above_threshold"]]
    n_false = len(false_activation_frames)
    duration_min = duration_s / 60.0
    rate_per_minute = n_false / duration_min

    output = {
        "config": {
            "threshold": THRESHOLD,
            "sklearn_version": sklearn.__version__,
            "cpu_model": "Intel(R) Core(TM) Ultra 7 256V",
            "python_version": platform.python_version(),
            "video_path": args.video_path,
            "video_fps": fps,
            "duration_seconds": duration_s,
            "total_frames": len(trace),
            "definition": "A false activation is any frame where the model predicts a "
                           "non-Rest class with confidence >= threshold, on footage "
                           "containing no deliberate gestures.",
        },
        "false_activation_frames": n_false,
        "false_activations_per_minute": rate_per_minute,
    }

    results_dir = os.path.join(ROOT, "results")
    os.makedirs(results_dir, exist_ok=True)
    out_path = os.path.join(results_dir, "false_activation.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"False activations: {n_false} frames over {duration_min:.2f} min "
          f"= {rate_per_minute:.3f} per minute")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
