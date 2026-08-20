import json
import os
import pickle
import platform

import numpy as np
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

SEED = 42
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

data_directory = pickle.load(open(os.path.join(ROOT, "data.pickle"), "rb"))
data = np.asarray(data_directory["data"])
labels = np.asarray(data_directory["labels"])
sessions = np.asarray(data_directory.get("sessions"))

if sessions is None or sessions[0] is None:
    raise SystemExit(
        "data.pickle has no 'sessions' field. Run processing_data.py (refactored for T4) "
        "against the data/sessions/<id>/<class>/ layout first."
    )

unique_sessions = sorted(set(sessions.tolist()))

results_dir = os.path.join(ROOT, "results")
os.makedirs(results_dir, exist_ok=True)
out_path = os.path.join(results_dir, "cross_session.json")

base_config = {
    "random_seed": SEED,
    "sklearn_version": sklearn.__version__,
    "numpy_version": np.__version__,
    "cpu_model": "Intel(R) Core(TM) Ultra 7 256V",
    "python_version": platform.python_version(),
    "feature_dimensionality": int(data.shape[1]),
    "sessions_found": unique_sessions,
}

if len(unique_sessions) < 2:
    output = {
        "config": base_config,
        "status": "blocked",
        "reason": (
            f"Only {len(unique_sessions)} session(s) found in data.pickle: {unique_sessions}. "
            "Cross-session evaluation requires at least two sessions captured under "
            "deliberately different conditions (lighting, background, distance, day). "
            "Run collect_session.py to record a second session, then re-run "
            "processing_data.py and this script."
        ),
        "within_session_accuracy": None,
        "cross_session_accuracy": None,
    }
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"BLOCKED: {output['reason']}")
    print(f"Wrote {out_path}")
    raise SystemExit(0)

# Within-session: standard stratified split on session A (first session) alone, same
# methodology as train_data.py, for direct comparison against the cross-session number.
session_a = unique_sessions[0]
session_b = unique_sessions[1]
if len(unique_sessions) > 2:
    print(f"More than 2 sessions found ({unique_sessions}); using '{session_a}' as train "
          f"and '{session_b}' as held-out test. Extra sessions are ignored by this run.")

mask_a = sessions == session_a
mask_b = sessions == session_b

x_a, y_a = data[mask_a], labels[mask_a]
x_b, y_b = data[mask_b], labels[mask_b]

x_train_within, x_test_within, y_train_within, y_test_within = train_test_split(
    x_a, y_a, test_size=0.2, shuffle=True, stratify=y_a, random_state=SEED
)
model_within = RandomForestClassifier(random_state=SEED)
model_within.fit(x_train_within, y_train_within)
within_pred = model_within.predict(x_test_within)
within_accuracy = float(accuracy_score(within_pred, y_test_within))
within_report = classification_report(y_test_within, within_pred, output_dict=True, zero_division=0)

# Cross-session: train on all of session A, evaluate on all of session B.
model_cross = RandomForestClassifier(random_state=SEED)
model_cross.fit(x_a, y_a)
cross_pred = model_cross.predict(x_b)
cross_accuracy = float(accuracy_score(cross_pred, y_b))
cross_report = classification_report(y_b, cross_pred, output_dict=True, zero_division=0)

output = {
    "config": {
        **base_config,
        "train_session": session_a,
        "test_session": session_b,
        "train_session_size": int(mask_a.sum()),
        "test_session_size": int(mask_b.sum()),
    },
    "status": "ok",
    "within_session": {
        "description": f"Stratified 80/20 split within session '{session_a}' only.",
        "accuracy": within_accuracy,
        "classification_report": within_report,
    },
    "cross_session": {
        "description": f"Trained on all of session '{session_a}', evaluated on all of session '{session_b}'.",
        "accuracy": cross_accuracy,
        "classification_report": cross_report,
    },
    "accuracy_delta": within_accuracy - cross_accuracy,
}

with open(out_path, "w") as f:
    json.dump(output, f, indent=2)

print(f"Within-session accuracy ({session_a}): {within_accuracy:.4f}")
print(f"Cross-session accuracy ({session_a} -> {session_b}): {cross_accuracy:.4f}")
print(f"Delta: {output['accuracy_delta']:.4f}")
print(f"Wrote {out_path}")
