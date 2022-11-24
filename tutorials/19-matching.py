import numpy as np
import cv2
from matplotlib import pyplot as plt

I1 = cv2.imread('images/box.jpg',0)
I2 = cv2.imread('images/box_in_scene.jpg',0)

# Using ORB and Brute force matching
# ==================================

# Initiate ORB
orb = cv2.ORB_create()
# Find keypoints and descriptors with ORB
keypoints1, descriptors1 = orb.detectAndCompute(I1, None)
keypoints2, descriptors2 = orb.detectAndCompute(I2, None)

# Create a brute-force matcher
bfmatcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
# Match the descriptors
matches = bfmatcher.match(descriptors1, descriptors2)
# Sort the matches in the order of their distance.
matches = sorted(matches, key = lambda x:x.distance)

# Draw first 10 matches.
I3 = cv2.drawMatches(I1, keypoints1, I2, keypoints2, matches[:10], None, flags=2)
# Plot the result with matplotlib
plt.imshow(I3), plt.show()

# Using SIFT and FLANN matching
# =============================

# Initiate SIFT
sift = cv2.xfeatures2d.SIFT_create()
# Find the keypoints and descriptors with SIFT
keypoints1, descriptors1 = sift.detectAndCompute(I1, None)
keypoints2, descriptors2 = sift.detectAndCompute(I2, None)

# FLANN parameters
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)   # Or an empty dictionary
flannmatcher = cv2.FlannBasedMatcher(index_params,search_params)
# Match the descriptors
matches = flannmatcher.knnMatch(descriptors1, descriptors2, k=2)

# Need to draw only good matches, so create a mask
matchesMask = [[0,0] for i in range(len(matches))]

# Ratio test
for i,(m,n) in enumerate(matches):
    if m.distance < 0.7*n.distance:
        matchesMask[i]=[1,0]

draw_params = dict(matchColor = (0,255,0),
                   singlePointColor = (255,0,0),
                   matchesMask = matchesMask,
                   flags = 0)
I3 = cv2.drawMatchesKnn(I1, keypoints1, I2, keypoints2, matches, None, **draw_params)
plt.imshow(I3,), plt.show()