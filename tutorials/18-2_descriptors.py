import cv2
import numpy as np

I = cv2.imread('images/blocks.jpg', 0)
#gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# SIFT
# ====
sift = cv2.xfeatures2d.SIFT_create()
keypoints = sift.detect(I, None)

# Draw keypoints
Isift = cv2.cvtColor(I, cv2.COLOR_GRAY2BGR)
cv2.drawKeypoints(I, keypoints, Isift, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

cv2.imshow('SIFT', Isift); cv2.waitKey(0)
#cv2.destroyAllWindows()

keypoints, descriptors = sift.detectAndCompute(I, None)
# What is the size of "descriptors"?


# # SURF
# # ====
# hessian_threshold = 4000
# surf = cv2.xfeatures2d.SURF_create(hessian_threshold)
# keypoints, descriptors = surf.detectAndCompute(I, None)
#
# # Draw keypoints
# Isurf = cv2.cvtColor(I, cv2.COLOR_GRAY2BGR)
# cv2.drawKeypoints(I, keypoints, Isurf, (255,0,0), 4)
#
# cv2.imshow('SURF', Isurf); cv2.waitKey(0)
# #cv2.destroyAllWindows()


# FAST
# ====
fast = cv2.FastFeatureDetector_create()
keypoints = fast.detect(I, None)

# Draw keypoints
Ifast = cv2.cvtColor(I, cv2.COLOR_GRAY2BGR)
cv2.drawKeypoints(I, keypoints, Ifast, color=(255,0,0))

cv2.imshow('FAST', Ifast); cv2.waitKey(0)
#cv2.destroyAllWindows()


# ORB
# ===
orb = cv2.ORB_create()
keypoints = orb.detect(I, None)
keypoints, descriptors = orb.compute(I, keypoints)

# Draw keypoints
Iorb = cv2.cvtColor(I, cv2.COLOR_GRAY2BGR)
cv2.drawKeypoints(I, keypoints, Iorb, color=(0,255,0), flags=0)

cv2.imshow('ORB', Iorb); cv2.waitKey(0)
cv2.destroyAllWindows()