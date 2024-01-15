import cv2
import numpy as np

from ultralytics import YOLO
model = YOLO("tugas/tugas3/coin.pt")

# video ver
cap = cv2.VideoCapture("tugas/tugas3/video/video2.mp4")

while True:
    ret, frame = cap.read()

    if not ret: break

    h, w, _ = frame.shape  
    resize = cv2.resize(frame, (int(w/2), int(h/2))) 

    results = model.predict(source=resize, save=False, save_txt=False, conf=0.5, verbose=False)

    for r in results:
        boxes = r.boxes

        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) 

            cv2.rectangle(resize, (x1, y1), (x2, y2), (255, 0, 255), 2)

    cv2.imshow('frame', resize)

    if cv2.waitKey(1) == 27 : break

cap.release()
cv2.destroyAllWindows()