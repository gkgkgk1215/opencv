import cv2
# import numpy as np

def OnMouse(x):
    pass

img = cv2.imread('example_images/brain_noise.jpeg')

cv2.namedWindow('BlurPane')
cv2.createTrackbar('BLUR_MODE', 'BlurPane', 0, 3, OnMouse)
cv2.createTrackbar('BLUR', 'BlurPane', 0, 6, OnMouse)

mode = cv2.getTrackbarPos('BLUR_MODE', 'BlurPane')
val = cv2.getTrackbarPos('BLUR', 'BlurPane')

while True:
    val = val*2 + 1
    try:
        if mode==0:
            blur = cv2.blur(img, (val, val))
            # blur = cv2.filter2D(img, -1, np.ones((val, val), np.float32)/(val**2))    # averaging filter
        elif mode==1:
            blur = cv2.GaussianBlur(img, (val,val), 0)
        elif mode==2:
            blur = cv2.medianBlur(img, val)
        elif mode==3:
            blur = cv2.bilateralFilter(img, val*5, val*15, val*15)
        else:
            break
        cv2.imshow('BlurPane', blur)
    except:
        break

    key = cv2.waitKey(1) & 0xFF
    if key==27:
        break

    mode = cv2.getTrackbarPos('BLUR_MODE', 'BlurPane')
    val = cv2.getTrackbarPos('BLUR', 'BlurPane')

cv2.destroyAllWindows()