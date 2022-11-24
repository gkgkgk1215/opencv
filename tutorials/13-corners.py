import cv2
import numpy as np

I = cv2.imread('images/blocks.jpg',0)

# Harris corners
neighborhood = 2  # neighborhood size
apperture = 3     # Apperture size
alpha = 0.04      # Parameter alpha (or k)
score = cv2.cornerHarris(I, neighborhood, apperture, alpha)
# Dilate only to show corners better
score = cv2.dilate(score,None)

# Show greater than 0.01*max corners in the image (as blue)
Ibgr = cv2.cvtColor(I, cv2.COLOR_GRAY2BGR)
Ibgr[score > 0.01*score.max()] = [255, 0, 0]

cv2.imshow('Harris Corners', Ibgr)
cv2.waitKey(0)
cv2.destroyAllWindows()


# Parameters for shi-Tomasi (goodFeaturesToTrack)
numcorners = 25  # Number of best corners to keep
quality = 0.01   # Reject below the quality value
mindist = 10     # Minimum euclidean distance between corners
# Corner detection
corners = cv2.goodFeaturesToTrack(I, numcorners, quality, mindist)
corners = np.int0(corners)

# Draw circles around the corners
Ibgr = cv2.cvtColor(I, cv2.COLOR_GRAY2BGR)
for k in corners:
    x,y = k.ravel()
    cv2.circle(Ibgr, (x,y), 3, 255, -1)

cv2.imshow('Shi-Tomasi', Ibgr)
cv2.waitKey(0)
cv2.destroyAllWindows()


