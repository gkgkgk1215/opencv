import cv2
import numpy as np

img_scene = cv2.imread('example_images/dgist_building.jpg')
img_logo = cv2.imread('example_images/dgist_logo2.jpg')    # logo

# Logo image resize
img_logo = cv2.resize(img_logo, None, fx=0.5, fy=0.5)

# original images
cv2.imshow("background image", img_scene)
cv2.imshow("logo image", img_logo)
cv2.waitKey(0)

# creating mask and mask_inverse
img_logo_gray = cv2.cvtColor(img_logo, cv2.COLOR_BGR2GRAY)
cv2.imshow("img_logo_gray", img_logo_gray)
cv2.waitKey(0)

# White (255): background, Black(0): logo part
ret, mask = cv2.threshold(img_logo_gray, 245, 255, cv2.THRESH_BINARY)
cv2.imshow("mask", mask)
cv2.waitKey(0)

# White (255): logo part, Black(0): background
mask_inv = cv2.bitwise_not(mask)
cv2.imshow("mask_inv", mask_inv)
cv2.waitKey(0)

# area selection to position the logo
h, w, _ = img_logo.shape
hpos = 620
vpos = 20
roi = img_scene[vpos:h+vpos, hpos:w+hpos]   # region of interest

# make logo black and abstract logo
img_logo_bg = cv2.bitwise_and(roi, roi, mask=mask)
cv2.imshow("img_logo_bg", img_logo_bg)
cv2.waitKey(0)

img_logo_fg = cv2.bitwise_and(img_logo, img_logo, mask=mask_inv)
cv2.imshow("img_logo_fg", img_logo_fg)
cv2.waitKey(0)

# make logo background transparent and add logo image
dst = cv2.add(img_logo_bg, img_logo_fg)
img_scene[vpos:vpos + h, hpos:hpos + w] = dst

cv2.imshow('result', img_scene)
cv2.waitKey(0)
cv2.destroyAllWindows()