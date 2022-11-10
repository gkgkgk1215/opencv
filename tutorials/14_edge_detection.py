import numpy as np
import cv2
import matplotlib.pyplot as plt

img = cv2.imread("example_images/butterfly.jpg", cv2.IMREAD_GRAYSCALE)

img_edge1 = cv2.Canny(img, threshold1=50, threshold2=300)
img_edge2 = cv2.Canny(img, threshold1=100, threshold2=300)
img_edge3 = cv2.Canny(img, threshold1=200, threshold2=300)

cv2.imshow("original", img)
cv2.imshow("edge1", img_edge1)
cv2.imshow("edge2", img_edge2)
cv2.imshow("edge3", img_edge3)
cv2.waitKey(0)
cv2.destroyAllWindows()
