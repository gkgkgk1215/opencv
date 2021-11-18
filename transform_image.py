import cv2
import numpy as np

img = cv2.imread("example_images/deer.jpg")

# Scaling
img_sc = cv2.resize(img, None, fx=0.5, fy=0.5)
cv2.imshow("Resized", img_sc)
cv2.waitKey(0)
Nrows, Ncols, channels = img_sc.shape

# Translation
tx = 100
ty = 50
M = np.float32([[1, 0, tx], [0, 1, ty]])
img_tr = cv2.warpAffine(img_sc, M, (Ncols, Nrows))
cv2.imshow("Translated", img_tr)
cv2.waitKey(0)

# Rotation
center_rot = (Ncols/2, Nrows/2)
angle = 45  # counter-clock wise
M = cv2.getRotationMatrix2D(center_rot, angle, scale=1)
img_rot = cv2.warpAffine(img_sc, M, (Ncols, Nrows))
cv2.imshow("Rotated", img_rot)
cv2.waitKey(0)