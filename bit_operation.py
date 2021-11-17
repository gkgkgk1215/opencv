import cv2
import numpy as np

img1 = cv2.imread('example_images/dgist_building.jpg')
img2 = cv2.imread('example_images/dgist_logo2.jpg')    # logo
h, w, _ = np.shape(img2)
img2 = cv2.resize(img2, (int(w/3), int(h/3)))   # image resize

# creating mask and mask_inverse
img2gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
cv2.imshow("img_gray", img2gray)
cv2.waitKey(0)

ret, mask = cv2.threshold(img2gray, 245, 255, cv2.THRESH_BINARY)
cv2.imshow("mask", mask)
cv2.waitKey(0)

mask_inv = cv2.bitwise_not(mask)
cv2.imshow("mask_inv", mask_inv)
cv2.waitKey(0)

# area selection to position the logo
h, w, channels = img2.shape
hpos = 10
vpos = 10
roi = img1[vpos:h+vpos, hpos:w+hpos]

# make logo black and abstract logo
img1_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
cv2.imshow("img1_bg", img1_bg)
cv2.waitKey(0)

img2_fg = cv2.bitwise_and(img2, img2, mask=mask)
cv2.imshow("img2_fg", img2_fg)
cv2.waitKey(0)

# make logo background transparent and add logo image
dst = cv2.add(img1_bg, img2_fg)
img1[vpos:vpos + h, hpos:hpos + w] = dst

cv2.imshow('result', img1)
cv2.waitKey(0)
cv2.destroyAllWindows()