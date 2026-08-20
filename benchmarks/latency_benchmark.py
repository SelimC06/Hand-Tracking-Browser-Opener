import json
import os
import pickle
import platform
import time

import cv2
import mediapipe as mp
import numpy as np
import sklearn
from sklearn.ensemble import RandomForestClassifier

SEED = 42
N_FRAMES = 1000
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

data_directory = pickle.load(open(os.path.join(ROOT, "data.pickle"), "rb"))
data = np.asarray(data_directory["data"])
labels = np.asarray(data_directory["labels"])

model = RandomForestClassifier(random_state=SEED)
model.fit(data, labels)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Could not open webcam (index 0)")

# Warm-up: let the camera settle and pay any one-time init cost outside the measured loop.
for _ in range(30):
    cap.read()

capture_ms = []
mediapipe_ms = []
feature_ms = []
predict_ms = []
hand_detected_count = 0

loop_start = time.perf_counter()
for i in range(N_FRAMES):
    t0 = time.perf_counter()
    ret, image = cap.read()
    t1 = time.perf_counter()
    capture_ms.append((t1 - t0) * 1000.0)
    if not ret:
        continue

    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)
    t2 = time.perf_counter()
    mediapipe_ms.append((t2 - t1) * 1000.0)

    if not results.multi_hand_landmarks:
        continue

    data_setup = []
    for handLms in results.multi_hand_landmarks:
        for lm in handLms.landmark:
            data_setup.append(lm.x)
            data_setup.append(lm.y)
    t3 = time.perf_counter()
    feature_ms.append((t3 - t2) * 1000.0)

    if len(data_setup) != 42:
        continue

    _ = model.predict_proba([np.asarray(data_setup)])
    t4 = time.perf_counter()
    predict_ms.append((t4 - t3) * 1000.0)

    hand_detected_count += 1

    if (i + 1) % 100 == 0:
        print(f"{i + 1}/{N_FRAMES} frames captured...")

loop_end = time.perf_counter()
cap.release()


def stats_ms(values):
    if not values:
        return None
    arr = np.asarray(values)
    return {
        "n": int(arr.size),
        "mean_ms": float(np.mean(arr)),
        "p50_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)),
    }


total_wall_seconds = loop_end - loop_start
sustained_fps = N_FRAMES / total_wall_seconds

output = {
    "config": {
        "random_seed": SEED,
        "sklearn_version": sklearn.__version__,
        "opencv_version": cv2.__version__,
        "mediapipe_version": mp.__version__,
        "numpy_version": np.__version__,
        "cpu_model": "Intel(R) Core(TM) Ultra 7 256V",
        "python_version": platform.python_version(),
        "frames_attempted": N_FRAMES,
        "frames_with_hand_detected": hand_detected_count,
        "hands_config": {"static_image_mode": True, "min_detection_confidence": 0.3},
        "note": "capture and mediapipe stages measured over every attempted frame; "
                "feature construction and predict_proba stages measured only over "
                "frames where a hand was detected with exactly 42 features.",
    },
    "stages_ms": {
        "frame_capture": stats_ms(capture_ms),
        "mediapipe_landmark_extraction": stats_ms(mediapipe_ms),
        "feature_vector_construction": stats_ms(feature_ms),
        "randomforest_predict_proba": stats_ms(predict_ms),
    },
    "end_to_end": {
        "total_wall_seconds": total_wall_seconds,
        "frames_attempted": N_FRAMES,
        "sustained_fps": sustained_fps,
    },
}

results_dir = os.path.join(ROOT, "results")
os.makedirs(results_dir, exist_ok=True)
out_path = os.path.join(results_dir, "latency.json")
with open(out_path, "w") as f:
    json.dump(output, f, indent=2)

print(f"Wrote {out_path}")
print(json.dumps(output["stages_ms"], indent=2))
print(f"Sustained FPS: {sustained_fps:.2f}")
