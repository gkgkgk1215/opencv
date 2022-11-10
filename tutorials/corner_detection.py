import numpy as np
import cv2

img = cv2.imread("example_images/boxes.jpg")
img_copy = img.copy()
img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_grey = np.float32(img_grey)

# Harris Corner Detection
dst = cv2.cornerHarris(img_grey, blockSize=6, ksize=3, k=0.04)
img_copy[dst > 0.01*dst.max()] = [0, 0, 255]
cv2.imshow("Harris_corner_detection", img_copy)
cv2.waitKey(0)

# Shi-Tomasi Corner Detection
img_copy = img.copy()
corners = cv2.goodFeaturesToTrack(img_grey, maxCorners=100, qualityLevel=0.01, minDistance=10)
corners = np.int0(corners)
for i in corners:
    x, y = i.ravel()
    cv2.circle(img_copy, (x, y), 3, 255, -1)

cv2.imshow("Shi-Tomashi Corner Detection", img_copy)
cv2.waitKey(0)
cv2.destroyAllWindows()