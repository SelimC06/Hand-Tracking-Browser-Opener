import cv2
import mediapipe as mp
import pickle
import numpy as np
import webbrowser
import time

model_d = pickle.load(open('./model.p', 'rb'))
model = model_d['model']

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mpDraw = mp.solutions.drawing_utils
mpDrawStyles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode = True, min_detection_confidence = 0.3)

labels = {0: 'Close', 1: 'Open', 2: 'Side'}
test = {frozenset(["Close", "Open"]): "https://github.com/SelimC06",
        frozenset(["Side", "Open"]): "https://Google.com",
        frozenset(["Side", "Close"]): "https://www.linkedin.com/in/selim-coskunuzer-023ab9270/"}

browser_vision = {"Github" : ["Close", "Open"],
                  "Google" : ["Side", "Open"],
                  "LinkedIn" : ["Side", "Close"]}

threshold = 0.65
curr = []
while True:
    if len(curr) == 2:
        curr = []
    ret, image = cap.read()
    data_setup = []
    x_ = []
    y_ = []

    h, w, _ = image.shape

    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    results = hands.process(img_rgb)
    if results.multi_hand_landmarks:
        #for handLms in results.multi_hand_landmarks:
        #    mpDraw.draw_landmarks(
        #        image,
        #        handLms,
        #        mp_hands.HAND_CONNECTIONS,
        #        mpDrawStyles.get_default_hand_landmarks_style(),
        #        mpDrawStyles.get_default_hand_connections_style()
        #    )
        
        for handLms in results.multi_hand_landmarks:
                for i in range(len(handLms.landmark)):
                    x = handLms.landmark[i].x
                    y = handLms.landmark[i].y
                    data_setup.append(x)
                    data_setup.append(y)
                    x_.append(x)
                    y_.append(y)
        
        if x_ and y_:
            x1 = int(min(x_) * w) - 10
            x2 = int(max(x_) * w) - 10

            y1 = int(min(y_) * h) - 10
            y2 = int(max(y_) * h) - 10
        
        cv2.rectangle(image, (x1, y1), (x2, y2), (0,0,0), 2)
        if len(data_setup) == 42:
            probability = model.predict_proba([np.asarray(data_setup)])
        
            if max(probability[0]) >= threshold:
                prediction = model.predict([np.asarray(data_setup)])
                prediction_character = labels[int(prediction[0])]
                cv2.putText(image, str(curr), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2, cv2.LINE_AA)
                if not prediction_character in curr:
                    curr.append(prediction_character)
            else:
                cv2.putText(image, (str(max(probability[0]))), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2, cv2.LINE_AA)

    cv2.putText(image, str(curr), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)
    num = 80
    for idx, val in browser_vision.items():
        cv2.putText(image, (idx + ": " + str(val)), (0, num), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1, cv2.LINE_AA)
        num += 30
    url = test.get(frozenset(curr))
    if url:
        webbrowser.open(url)
        time.sleep(2)
    cv2.imshow("Image", image)
    cv2.waitKey(25)

cap.release()
cv2.destroyAllWindows()