import numpy as np
import cv2

img = cv2.imread("example_images/box_matching.png")
img = cv2.resize(img, None, fx=2.0, fy=2.0, )
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


# pip install opencv-contrib-python
# SIFT
sift = cv2.xfeatures2d.SIFT_create()    # create an SIFT object
kp = sift.detect(img_gray, None)    # extract key points
img2 = None
img2 = cv2.drawKeypoints(img_gray, kp, img2, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imshow('SIFT', img2)
cv2.waitKey(0)

# SURF
# surf = cv2.xfeatures2d.SURF_create()
# surf.setHessianThreshold(10000)
# kp, des = surf.detectAndCompute(img, None)
# img = cv2.drawKeypoints(img_gray, kp, None, (255, 0, 0), 4)
# cv2.imshow('img', img)
# cv2.waitKey(0)

# FAST
fast = cv2.FastFeatureDetector_create(30)
kp = fast.detect(img, None)
img2 = None
img2 = cv2.drawKeypoints(img, kp, img2, (255, 0, 0))
cv2.imshow('FAST1', img2)
cv2.waitKey(0)

fast.setNonmaxSuppression(0)
kp = fast.detect(img, None)
img2 = None
img2 = cv2.drawKeypoints(img, kp, img2, (255, 0, 0))
cv2.imshow('FAST2', img2)
cv2.waitKey(0)

# BRIEF
star = cv2.xfeatures2d.StarDetector_create()    # STAR detector
brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()   # BRIEF extractor
kp1 = star.detect(img, None)    # detecting using STAR
kp2, des = brief.compute(img, kp1) # calculating descriptors using BRIEF
img2 = None
img2 = cv2.drawKeypoints(img, kp1, img2, (255, 0, 0))
cv2.imshow('BRIEF', img2)
cv2.waitKey(0)

# ORB (not free, patented by OpenCV Labs)
orb = cv2.ORB_create()
kp, des = orb.detectAndCompute(img, None)
img2 = None
img2 = cv2.drawKeypoints(img, kp, img2, (0, 0, 255), flags=0)
cv2.imshow("ORB", img2)
cv2.waitKey(0)
