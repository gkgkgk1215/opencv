import numpy as np
import cv2
from matplotlib import pyplot as plt

img_org = cv2.imread('example_images/playing_card.jpg', cv2.IMREAD_GRAYSCALE)
img_tmpl = cv2.imread('example_images/card_template.png', cv2.IMREAD_GRAYSCALE)
cv2.imshow("playing_card", img_org)
cv2.imshow("template", img_tmpl)
cv2.waitKey(0)

h, w = np.shape(img_tmpl)
methods = ['cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR', 'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']

for m in methods:
    img = img_org.copy()
    method = eval(m)
    res = cv2.matchTemplate(img_org, img_tmpl, method)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        top_left = min_loc
    else:
        top_left = max_loc

    bottom_right = (top_left[0]+w, top_left[1]+h)
    cv2.rectangle(img, top_left, bottom_right, 255, 2)

    plt.subplot(121)
    plt.imshow(res, cmap='gray')
    plt.title('Matching Result')
    plt.xticks([])
    plt.yticks([])

    plt.subplot(122)
    plt.imshow(img, cmap='gray')
    plt.title('Detected Point')
    plt.xticks([])
    plt.yticks([])
    plt.title(m)
    plt.show()
