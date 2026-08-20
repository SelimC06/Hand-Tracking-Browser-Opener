import mediapipe as mp
import pickle
import cv2
import os

SESSIONS_DIR = os.path.join(os.path.dirname(__file__), "data", "sessions")

mp_hands = mp.solutions.hands
mpDraw = mp.solutions.drawing_utils
mpDrawStyles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

data = []
label_data = []
session_data = []

for session_id in sorted(os.listdir(SESSIONS_DIR)):
    session_path = os.path.join(SESSIONS_DIR, session_id)
    if not os.path.isdir(session_path):
        continue

    for class_dir in sorted(os.listdir(session_path)):
        class_path = os.path.join(session_path, class_dir)
        if not os.path.isdir(class_path):
            continue

        for img_path in os.listdir(class_path):
            data_setup = []
            img = cv2.imread(os.path.join(class_path, img_path))
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            results = hands.process(img_rgb)
            if results.multi_hand_landmarks:
                for handLms in results.multi_hand_landmarks:
                    for i in range(len(handLms.landmark)):
                        x = handLms.landmark[i].x
                        y = handLms.landmark[i].y
                        data_setup.append(x)
                        data_setup.append(y)
                data.append(data_setup)
                label_data.append(class_dir)
                session_data.append(session_id)

pickle_path = os.path.join(os.path.dirname(__file__), "data.pickle")
with open(pickle_path, "wb") as f:
    pickle.dump({"data": data, "labels": label_data, "sessions": session_data}, f)
print(f"Wrote pickle to: {pickle_path}")
print(f"Sessions found: {sorted(set(session_data))}")
