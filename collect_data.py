import cv2
import os

Data_dir = './data'
if not os.path.exists(Data_dir):
    os.makedirs(Data_dir)

num_classes = 3
dataset_size = 200

cap = cv2.VideoCapture(0)

for i in range(num_classes):
    if not os.path.exists(os.path.join(Data_dir, str(i))):
        os.makedirs(os.path.join(Data_dir, str(i)))

    print(f"Collecting data for {i}")

    while True:
        success, image = cap.read()
        cv2.putText(image, 'Ready? Press "Q" !', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 0, 0), 3, cv2.LINE_AA)
        cv2.imshow("image", image)
        if cv2.waitKey(25) == ord('q'):
            break
    
    counter = 0
    while counter < dataset_size:
        ret, image = cap.read()
        cv2.imshow('image', image)
        cv2.waitKey(25)
        cv2.imwrite(os.path.join(Data_dir, str(i), f'{counter}.jpg'), image)

        counter += 1

cap.release()
cv2.destroyAllWindows()
