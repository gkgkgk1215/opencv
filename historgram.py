import numpy as np
import cv2
from matplotlib import pyplot as plt

# Load an image as gray
img_gray = cv2.imread('example_images/dgist_building.jpg', cv2.IMREAD_GRAYSCALE)
cv2.imshow("img_gray", img_gray)
cv2.waitKey(0)

# Compute the histogram:
hist = cv2.calcHist(images=[img_gray], channels=[0], mask=None, histSize=[256], ranges=[0, 256])

# Plot the histogram using matplotlib
plt.plot(hist)
plt.show()


# Histogram Equalization
img_eq = cv2.equalizeHist(img_gray)
cv2.imshow("equalized", img_eq)
cv2.waitKey(0)

# Plot the equalized histogram
hist_eq = cv2.calcHist(images=[img_eq], channels=[0], mask=None, histSize=[256], ranges=[0, 256])
plt.plot(hist_eq)
plt.show()