import cv2
import numpy as np

def onChange(x):
    pass

img = cv2.imread('example_images/dgist_building.jpg', cv2.IMREAD_GRAYSCALE)
img_thr = np.zeros_like(img)
cv2.namedWindow(winname='thresholding_windows')
cv2.createTrackbar('tr', 'thresholding_windows', 0, 255, onChange)

while True:
    cv2.imshow('thresholding_windows', img_thr)
    key = cv2.waitKey(1) & 0xFF     # ESC key
    if key == 27:
        break
    thr_value = cv2.getTrackbarPos('tr', 'thresholding_windows')
    ret, img_thr = cv2.threshold(img, thr_value, 255, cv2.THRESH_BINARY)
cv2.destroyAllWindows()