import json
import os
import pickle

import cv2
import mediapipe as mp
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REST_LABEL = "3"


def load_model():
    with open(os.path.join(ROOT, "model.p"), "rb") as f:
        model_d = pickle.load(f)
    return model_d["model"]


def require_rest_class(model):
    if REST_LABEL not in list(model.classes_):
        raise SystemExit(
            f"model.p was not trained with a Rest class (classes_ = {list(model.classes_)}). "
            "Capture Rest-class images with collect_session.py, regenerate data.pickle with "
            "processing_data.py, and retrain with train_data.py before running this benchmark. "
            "Without a Rest class, the model is forced to pick among gesture classes for every "
            "frame and the false-activation rate would be meaningless."
        )


def process_footage(video_path, model, threshold=0.65, min_detection_confidence=0.3):
    """Run mediapipe + classifier over every frame of a footage file.

    Returns a list of per-frame dicts: frame_idx, timestamp_s, predicted_label,
    confidence, is_gesture_above_threshold (True if the model predicted a non-Rest
    class with confidence >= threshold).
    """
    if not os.path.exists(video_path):
        raise SystemExit(f"Footage file not found: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise SystemExit(f"Could not open footage file: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=min_detection_confidence)

    trace = []
    frame_idx = 0
    while True:
        ret, image = cap.read()
        if not ret:
            break

        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        predicted_label = None
        confidence = None
        is_gesture_above_threshold = False

        if results.multi_hand_landmarks:
            data_setup = []
            for handLms in results.multi_hand_landmarks:
                for lm in handLms.landmark:
                    data_setup.append(lm.x)
                    data_setup.append(lm.y)
            if len(data_setup) == 42:
                proba = model.predict_proba([np.asarray(data_setup)])[0]
                class_order = model.classes_
                idx = int(np.argmax(proba))
                predicted_label = str(class_order[idx])
                confidence = float(proba[idx])
                is_gesture_above_threshold = (
                    predicted_label != REST_LABEL and confidence >= threshold
                )

        trace.append({
            "frame_idx": frame_idx,
            "timestamp_s": frame_idx / fps,
            "predicted_label": predicted_label,
            "confidence": confidence,
            "is_gesture_above_threshold": is_gesture_above_threshold,
        })
        frame_idx += 1

    cap.release()
    return trace, fps


def trace_cache_path(video_path, threshold):
    base = os.path.splitext(os.path.basename(video_path))[0]
    cache_dir = os.path.join(ROOT, "results", "_trace_cache")
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, f"{base}_t{threshold}.json")


def get_or_build_trace(video_path, model, threshold=0.65):
    cache_path = trace_cache_path(video_path, threshold)
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            cached = json.load(f)
        return cached["trace"], cached["fps"]

    trace, fps = process_footage(video_path, model, threshold=threshold)
    with open(cache_path, "w") as f:
        json.dump({"video_path": video_path, "threshold": threshold, "fps": fps, "trace": trace}, f)
    return trace, fps
