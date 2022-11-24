import cv2
import numpy as np
from matplotlib import pyplot as plt

I = cv2.imread('images/kittens.jpg',0)
template = cv2.imread('images/cface.jpg',0)
width, height = template.shape

method = cv2.TM_CCOEFF_NORMED
#method = cv2.TM_CCORR_NORMED

# Apply template Matching
Iscores = cv2.matchTemplate(I, template, method)
# Extract the location
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(Iscores)
top_left = max_loc
bottom_right = (top_left[0]+width, top_left[1]+height)

# Draw a rectangle on the part of the image that matches 
cv2.rectangle(I, top_left, bottom_right, 255, 2)

plt.subplot(121), plt.imshow(Iscores, cmap = 'gray')
plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(I, cmap = 'gray')
plt.title('Detected Object'), plt.xticks([]), plt.yticks([])
plt.show()

# Detect multiple matchings
I = cv2.imread('images/kittens.jpg',0)
threshold = 0.55
location = np.where(Iscores >= threshold)
for pt in zip(*location[::-1]):
    cv2.rectangle(I, pt, (pt[0]+width, pt[1]+height), (0,0,255), 2)
cv2.imshow('result', I)
cv2.waitKey(0)
