import json
import os
import platform

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pickle
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

SEED = 42
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

data_directory = pickle.load(open(os.path.join(ROOT, "data.pickle"), "rb"))
data = np.asarray(data_directory["data"])
labels = np.asarray(data_directory["labels"])

x_train, x_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.2, shuffle=True, stratify=labels, random_state=SEED
)

model = RandomForestClassifier(random_state=SEED)
model.fit(x_train, y_train)

proba = model.predict_proba(x_test)
class_order = model.classes_
predicted_idx = np.argmax(proba, axis=1)
predicted_label = class_order[predicted_idx]
max_proba = proba[np.arange(len(proba)), predicted_idx]
correct = predicted_label == y_test

thresholds = [round(t, 2) for t in np.arange(0.30, 0.95 + 1e-9, 0.05)]

sweep = []
n_total = len(y_test)
for t in thresholds:
    accepted_mask = max_proba >= t
    n_accepted = int(accepted_mask.sum())
    n_rejected = n_total - n_accepted
    rejection_rate = n_rejected / n_total

    if n_accepted > 0:
        precision_on_accepted = float(correct[accepted_mask].sum() / n_accepted)
    else:
        precision_on_accepted = None

    n_correct_accepted = int((correct & accepted_mask).sum())
    effective_accuracy = n_correct_accepted / n_total

    sweep.append({
        "threshold": t,
        "n_accepted": n_accepted,
        "n_rejected": n_rejected,
        "rejection_rate": rejection_rate,
        "precision_on_accepted": precision_on_accepted,
        "effective_accuracy_over_all_frames": effective_accuracy,
    })

shipped_threshold = 0.65
shipped_row = next(r for r in sweep if r["threshold"] == shipped_threshold)

output = {
    "config": {
        "random_seed": SEED,
        "sklearn_version": sklearn.__version__,
        "numpy_version": np.__version__,
        "cpu_model": "Intel(R) Core(TM) Ultra 7 256V",
        "python_version": platform.python_version(),
        "feature_dimensionality": int(data.shape[1]),
        "test_set_size": n_total,
        "class_counts_test": {str(c): int((y_test == c).sum()) for c in sorted(set(labels.tolist()))},
        "shipped_threshold": shipped_threshold,
        "note": "Swept on a within-session held-out split (same session as training data); "
                "see results/cross_session.json (T4) for a non-leaked estimate once available.",
    },
    "sweep": sweep,
    "shipped_threshold_row": shipped_row,
}

results_dir = os.path.join(ROOT, "results")
os.makedirs(results_dir, exist_ok=True)

with open(os.path.join(results_dir, "threshold_sweep.json"), "w") as f:
    json.dump(output, f, indent=2)

thresholds_arr = [r["threshold"] for r in sweep]
precision_arr = [r["precision_on_accepted"] for r in sweep]
rejection_arr = [r["rejection_rate"] for r in sweep]
effective_arr = [r["effective_accuracy_over_all_frames"] for r in sweep]

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(thresholds_arr, precision_arr, marker="o", label="Precision on accepted predictions")
ax.plot(thresholds_arr, rejection_arr, marker="s", label="Rejection rate")
ax.plot(thresholds_arr, effective_arr, marker="^", label="Effective accuracy (all frames)")
ax.axvline(shipped_threshold, color="gray", linestyle="--", label=f"Shipped threshold ({shipped_threshold})")
ax.set_xlabel("predict_proba confidence threshold")
ax.set_ylabel("rate")
ax.set_ylim(-0.02, 1.02)
ax.set_title("Confidence-threshold sweep (within-session held-out split)")
ax.legend(loc="lower left", fontsize=8)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(results_dir, "threshold_sweep.png"), dpi=150)

print(f"Wrote {os.path.join(results_dir, 'threshold_sweep.json')}")
print(f"Wrote {os.path.join(results_dir, 'threshold_sweep.png')}")
print(f"Shipped threshold {shipped_threshold}: {shipped_row}")
