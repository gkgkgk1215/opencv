import numpy as np
import cv2

img = np.zeros((512, 1024, 3), np.uint8)

# drawing lines with various color and thickness
# BGR: Blue-Green-Red order
cv2.line(img=img,pt1=(0, 0), pt2=(400, 100), color=(255, 0, 0), thickness=5)
# cv2.triangle()
cv2.rectangle(img, (384, 0), (510, 128), (0, 255, 0), 3)
cv2.circle(img, (447, 63), 63, (0, 0, 255), -1)
cv2.ellipse(img, (256, 256), (100, 50), 20, 45, 270, (255, 0, 0), -1)
font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(img, 'Robot System Design', (10, 500), font, 3, (255, 255, 255), 2)

cv2.imshow('drawing', img)
cv2.waitKey(0)
cv2.destroyAllWindows()