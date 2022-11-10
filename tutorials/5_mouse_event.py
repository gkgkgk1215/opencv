import numpy as np
import cv2
from random import shuffle

def onMouse(event, x,y, flags, param):
    global radius
    if event == cv2.EVENT_LBUTTONDOWN:
        shuffle(b), shuffle(g), shuffle(r)
        cv2.circle(param, (x, y), radius, (b[0], g[0], r[0]), -1)
    elif event == cv2.EVENT_MOUSEWHEEL:
        if flags > 0:
            radius -= 1
        else:
            radius += 1

b = [i for i in range(256)]
g = [i for i in range(256)]
r = [i for i in range(256)]
radius = 20
img = np.zeros((512,512,3), np.uint8)
cv2.namedWindow('mouseEvent')
cv2.setMouseCallback('mouseEvent', onMouse, param=img)

while True:
    cv2.imshow('mouseEvent', img)
    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break
cv2.destroyAllWindows()

