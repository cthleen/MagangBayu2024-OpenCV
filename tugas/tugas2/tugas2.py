import cv2
import numpy as np

img = cv2.imread("tugas/tugas2/tugas2.jpg")

cv2.imshow("Gambar", img)

img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

lower_range = np.array([90, 100, 100])
upper_range = np.array([130, 255, 255])

mask = cv2.inRange(img_hsv, lower_range, upper_range)

contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

lineheight = 11

for contour in contours:
    epsilon = 0.02 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)

    if len(approx) == 4:
       x, y, w, h = cv2.boundingRect(approx)
       cropped = img[y+lineheight:y+h-lineheight, x+lineheight:x+w-lineheight]

canny = cv2.Canny(cropped, 50, 150)

contours2, _ = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

for c in contours2:
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)
    sides = len(approx)
    print(sides)

cv2.putText(img, str(sides), (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 0), 2)

cv2.drawContours(cropped, contours2, -1, (0, 0, 0), 3)

cv2.imshow("Gambar2", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
