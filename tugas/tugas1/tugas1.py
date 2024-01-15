import cv2
import numpy as np

img = cv2.imread("tugas/tugas1/tugas1.png")

cv2.imshow("Gambar", img)

img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

lower_range = np.array([30, 100, 100])
upper_range = np.array([90, 255, 255])

mask = cv2.inRange(img_hsv, lower_range, upper_range)

contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

min = 100

for contour in contours:
    area = cv2.contourArea(contour)
    if area > min:
        x, y, w, h = cv2.boundingRect(contour)

        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)

cv2.imshow("Gambar2", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

