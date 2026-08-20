import json
import os
import pickle
import platform

import numpy as np
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

SEED = 42

data_directory = pickle.load(open("./data.pickle", 'rb'))

data = np.asarray(data_directory['data'])
labels = np.asarray(data_directory['labels'])

x_train, x_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.2, shuffle=True, stratify=labels, random_state=SEED
)

model_params = {"random_state": SEED}
model = RandomForestClassifier(**model_params)

model.fit(x_train, y_train)

y_predict = model.predict(x_test)

score = accuracy_score(y_predict, y_test)

print(f"{score*100}% of samples were classified correctly !")

pickle_path = os.path.join(os.path.dirname(__file__), "model.p")
with open(pickle_path, "wb") as f:
    pickle.dump({"model": model}, f)
print(f"Wrote pickle to: {pickle_path}")

class_labels = sorted(set(labels.tolist()))
report = classification_report(y_test, y_predict, labels=class_labels, output_dict=True, zero_division=0)
cm = confusion_matrix(y_test, y_predict, labels=class_labels)

class_counts_total = {cls: int((labels == cls).sum()) for cls in class_labels}
class_counts_train = {cls: int((y_train == cls).sum()) for cls in class_labels}
class_counts_test = {cls: int((y_test == cls).sum()) for cls in class_labels}

metrics = {
    "config": {
        "random_seed": SEED,
        "sklearn_version": sklearn.__version__,
        "numpy_version": np.__version__,
        "cpu_model": "Intel(R) Core(TM) Ultra 7 256V",
        "python_version": platform.python_version(),
        "feature_dimensionality": int(data.shape[1]),
        "total_samples": int(data.shape[0]),
        "class_counts_total": class_counts_total,
        "class_counts_train": class_counts_train,
        "class_counts_test": class_counts_test,
        "test_size": 0.2,
        "model_hyperparameters": model.get_params(),
    },
    "results": {
        "overall_accuracy": float(score),
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_labels": class_labels,
    },
}

results_dir = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(results_dir, exist_ok=True)
metrics_path = os.path.join(results_dir, "classifier_metrics.json")
with open(metrics_path, "w") as f:
    json.dump(metrics, f, indent=2)
print(f"Wrote metrics to: {metrics_path}")
