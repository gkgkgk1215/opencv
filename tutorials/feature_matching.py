import numpy as np
import cv2

img1 = cv2.imread("example_images/box_matching.png", cv2.COLOR_BGR2GRAY)
# img1 = cv2.resize(img1, None, fx=1.0, fy=2.0)
img2 = cv2.imread("example_images/box_in_scene.png", cv2.COLOR_BGR2GRAY)
res = None

orb = cv2.ORB_create()
kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)

bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1, des2)

matches = sorted(matches, key=lambda x:x.distance)
res = cv2.drawMatches(img1, kp1, img2, kp2, matches[:20], res, flags=0)

cv2.imshow("Feature Matching", res)
cv2.waitKey(0)
