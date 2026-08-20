import json
import os
from datetime import datetime

import cv2

DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "sessions")
CLASS_NAMES = {0: "Close", 1: "Open", 2: "Side", 3: "Rest"}
DATASET_SIZE = 200


def prompt(msg):
    return input(msg).strip()


def main():
    session_id = prompt("Session ID (e.g. session_2): ")
    session_dir = os.path.join(DATA_DIR, session_id)
    if os.path.exists(session_dir):
        raise SystemExit(f"Session '{session_id}' already exists at {session_dir}")
    os.makedirs(session_dir)

    lighting = prompt("Lighting description (e.g. 'overhead fluorescent, evening'): ")
    background = prompt("Background description (e.g. 'plain white wall'): ")
    distance = prompt("Approx. distance from camera (e.g. '60cm'): ")
    notes = prompt("Any other notes (optional): ")

    meta = {
        "session_id": session_id,
        "date_range": {
            "earliest": datetime.now().isoformat(timespec="seconds"),
            "latest": None,
            "source": "recorded by collect_session.py at capture time",
        },
        "lighting": lighting,
        "background": background,
        "distance": distance,
        "notes": notes,
        "class_counts": {},
    }

    cap = cv2.VideoCapture(0)

    for class_id, class_name in CLASS_NAMES.items():
        class_dir = os.path.join(session_dir, str(class_id))
        os.makedirs(class_dir, exist_ok=True)

        hint = " (relaxed/resting hand, no deliberate gesture)" if class_name == "Rest" else ""
        print(f"Collecting data for class {class_id} ({class_name}){hint}")
        while True:
            success, image = cap.read()
            cv2.putText(image, f'Ready for "{class_name}"? Press "Q" !', (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 3, cv2.LINE_AA)
            cv2.imshow("image", image)
            if cv2.waitKey(25) == ord('q'):
                break

        counter = 0
        while counter < DATASET_SIZE:
            ret, image = cap.read()
            cv2.imshow('image', image)
            cv2.waitKey(25)
            cv2.imwrite(os.path.join(class_dir, f'{counter}.jpg'), image)
            counter += 1

        meta["class_counts"][str(class_id)] = counter

    cap.release()
    cv2.destroyAllWindows()

    meta["date_range"]["latest"] = datetime.now().isoformat(timespec="seconds")

    with open(os.path.join(session_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Wrote session manifest to {os.path.join(session_dir, 'meta.json')}")


if __name__ == "__main__":
    main()
