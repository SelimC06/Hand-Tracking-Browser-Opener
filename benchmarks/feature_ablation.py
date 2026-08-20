import json
import os
import pickle
import platform

import numpy as np
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from feature_representations import REPRESENTATIONS

SEED = 42
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

data_directory = pickle.load(open(os.path.join(ROOT, "data.pickle"), "rb"))
raw_data = data_directory["data"]
labels = np.asarray(data_directory["labels"])
sessions = np.asarray(data_directory.get("sessions"))

results_dir = os.path.join(ROOT, "results")
os.makedirs(results_dir, exist_ok=True)
out_path = os.path.join(results_dir, "feature_ablation.json")

base_config = {
    "random_seed": SEED,
    "sklearn_version": sklearn.__version__,
    "numpy_version": np.__version__,
    "cpu_model": "Intel(R) Core(TM) Ultra 7 256V",
    "python_version": platform.python_version(),
}

if sessions is None or sessions[0] is None:
    raise SystemExit("data.pickle has no 'sessions' field; run the T4-refactored processing_data.py first.")

unique_sessions = sorted(set(sessions.tolist()))

if len(unique_sessions) < 2:
    output = {
        "config": {**base_config, "sessions_found": unique_sessions},
        "status": "blocked",
        "reason": (
            f"Only {len(unique_sessions)} session(s) found: {unique_sessions}. Feature "
            "representation ablation is evaluated on the T4 cross-session split, so it "
            "needs a second session recorded under different conditions first. Run "
            "collect_session.py, then processing_data.py, then cross_session_eval.py, "
            "then this script."
        ),
        "representations": None,
    }
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"BLOCKED: {output['reason']}")
    print(f"Wrote {out_path}")
    raise SystemExit(0)

session_a, session_b = unique_sessions[0], unique_sessions[1]
mask_a = sessions == session_a
mask_b = sessions == session_b

y_a, y_b = labels[mask_a], labels[mask_b]
raw_a = [raw_data[i] for i in range(len(raw_data)) if mask_a[i]]
raw_b = [raw_data[i] for i in range(len(raw_data)) if mask_b[i]]

# If session A has fewer classes than session B (e.g. B includes a Rest class A
# never saw), blended cross-session accuracy conflates "model never trained on
# this class" with "model doesn't generalize." Restrict to classes present in
# both sessions so the three representations are compared on equal footing.
classes_in_a = set(y_a.tolist())
shared_class_mask = np.isin(y_b, list(classes_in_a))
shared_classes = sorted(classes_in_a)
has_unseen_test_classes = not shared_class_mask.all()

representation_results = {}
for name, transform in REPRESENTATIONS.items():
    x_a = np.asarray([transform(v) for v in raw_a])
    x_b = np.asarray([transform(v) for v in raw_b])

    model = RandomForestClassifier(random_state=SEED)
    model.fit(x_a, y_a)
    pred_b = model.predict(x_b)
    cross_accuracy = float(accuracy_score(pred_b, y_b))
    cross_accuracy_shared = (
        float(accuracy_score(pred_b[shared_class_mask], y_b[shared_class_mask]))
        if has_unseen_test_classes else None
    )

    representation_results[name] = {
        "feature_dimensionality": int(x_a.shape[1]),
        "cross_session_accuracy": cross_accuracy,
        "cross_session_accuracy_shared_classes_only": cross_accuracy_shared,
    }

baseline_key = "cross_session_accuracy_shared_classes_only" if has_unseen_test_classes else "cross_session_accuracy"
baseline = representation_results["raw_coordinates"][baseline_key]
for name, entry in representation_results.items():
    entry["accuracy_delta_vs_raw"] = entry[baseline_key] - baseline

output = {
    "config": {
        **base_config,
        "sessions_found": unique_sessions,
        "train_session": session_a,
        "test_session": session_b,
    },
    "status": "ok",
    "shared_classes": shared_classes,
    "comparison_basis": baseline_key,
    "representations": representation_results,
}

with open(out_path, "w") as f:
    json.dump(output, f, indent=2)

print(f"Comparison basis: {baseline_key}")
for name, entry in representation_results.items():
    print(f"{name}: dim={entry['feature_dimensionality']} "
          f"{baseline_key}={entry[baseline_key]:.4f} "
          f"delta_vs_raw={entry['accuracy_delta_vs_raw']:+.4f}")
print(f"Wrote {out_path}")
