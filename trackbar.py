import cv2
import numpy as np

def onChange(x):
    pass

img = np.zeros((200,512,3), np.uint8)
cv2.namedWindow('color_palette')

cv2.createTrackbar('B', 'color_palette', 0, 255, onChange)
cv2.createTrackbar('G', 'color_palette', 0, 255, onChange)
cv2.createTrackbar('R', 'color_palette', 0, 255, onChange)

while True:
    cv2.imshow('color_palette', img)
    key = cv2.waitKey(1) & 0xFF
    if key==27:
        break

    b = cv2.getTrackbarPos('B', 'color_palette')
    g = cv2.getTrackbarPos('G', 'color_palette')
    r = cv2.getTrackbarPos('R', 'color_palette')
    img[:] = [b,g,r]
cv2.destroyAllWindows()