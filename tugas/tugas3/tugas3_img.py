import cv2
import numpy as np

from ultralytics import YOLO
model = YOLO("tugas/tugas3/coin.pt")

# image ver
img = cv2.imread("tugas/tugas3/img/img.jpeg") 
# img = cv2.imread("tugas/tugas3/img/img2.jpeg") 

results = model.predict(source=img, save=False, save_txt=False, conf=0.5, verbose=False)
for r in results:
    boxes = r.boxes

for box in boxes:
    x1, y1, x2, y2 = box.xyxy[0]
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) 

    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 2)

cv2.imshow('img', img)

cv2.waitKey(0)
cv2.destroyAllWindows()