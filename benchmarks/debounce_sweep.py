import argparse
import json
import os

from _footage_utils import ROOT, get_or_build_trace, load_model, require_rest_class

THRESHOLD = 0.65
MIN_FOOTAGE_SECONDS = 10 * 60
K_VALUES = list(range(1, 9))


def count_activations(trace, k):
    """Count debounced activation events: K consecutive frames with the same
    non-Rest predicted label, each above threshold. Overlapping windows are not
    double-counted -- the run counter resets once an event fires."""
    events = 0
    run_label = None
    run_len = 0
    for frame in trace:
        if frame["is_gesture_above_threshold"]:
            label = frame["predicted_label"]
            if label == run_label:
                run_len += 1
            else:
                run_label = label
                run_len = 1
            if run_len >= k:
                events += 1
                run_label = None
                run_len = 0
        else:
            run_label = None
            run_len = 0
    return events


def main():
    parser = argparse.ArgumentParser(description="Sweep debounce K (consecutive agreeing predictions) "
                                                   "against false-activation rate and added latency.")
    parser.add_argument("video_path", help="Path to the same >=10 minute non-gesturing footage used by false_activation.py")
    args = parser.parse_args()

    model = load_model()
    require_rest_class(model)

    trace, fps = get_or_build_trace(args.video_path, model, threshold=THRESHOLD)

    duration_s = trace[-1]["timestamp_s"] if trace else 0.0
    if duration_s < MIN_FOOTAGE_SECONDS:
        raise SystemExit(
            f"Footage is only {duration_s:.1f}s long; the task requires at least "
            f"{MIN_FOOTAGE_SECONDS}s (10 minutes). Record more footage first."
        )
    duration_min = duration_s / 60.0

    latency_path = os.path.join(ROOT, "results", "latency.json")
    if not os.path.exists(latency_path):
        raise SystemExit(f"{latency_path} not found. Run benchmarks/latency_benchmark.py (T3) first.")
    with open(latency_path) as f:
        latency_data = json.load(f)
    stages = latency_data["stages_ms"]
    frame_time_ms = (
        stages["frame_capture"]["mean_ms"]
        + stages["mediapipe_landmark_extraction"]["mean_ms"]
        + stages["feature_vector_construction"]["mean_ms"]
        + stages["randomforest_predict_proba"]["mean_ms"]
    )

    sweep = []
    for k in K_VALUES:
        events = count_activations(trace, k)
        rate_per_minute = events / duration_min
        added_latency_ms = k * frame_time_ms
        sweep.append({
            "k": k,
            "false_activation_events": events,
            "false_activations_per_minute": rate_per_minute,
            "added_trigger_latency_ms": added_latency_ms,
        })

    # Operating point: smallest K that drives false activations to zero on this
    # footage; falls back to the largest K swept if none reaches zero.
    zero_rate = [row for row in sweep if row["false_activation_events"] == 0]
    chosen = zero_rate[0] if zero_rate else sweep[-1]
    chosen_reason = (
        f"smallest K in the sweep with zero measured false activations on {os.path.basename(args.video_path)}"
        if zero_rate else
        f"no K in [1,8] reached zero false activations on this footage; K={chosen['k']} "
        "is the largest swept, i.e. the most conservative option tested"
    )

    output = {
        "config": {
            "threshold": THRESHOLD,
            "video_path": args.video_path,
            "duration_seconds": duration_s,
            "frame_time_ms_source": "results/latency.json (sum of all four stage means)",
            "frame_time_ms": frame_time_ms,
        },
        "sweep": sweep,
        "chosen_k": chosen["k"],
        "chosen_k_reason": chosen_reason,
    }

    results_dir = os.path.join(ROOT, "results")
    os.makedirs(results_dir, exist_ok=True)
    out_path = os.path.join(results_dir, "debounce_sweep.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    for row in sweep:
        print(row)
    print(f"Chosen K = {chosen['k']} ({chosen_reason})")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
