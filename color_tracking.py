import cv2
import numpy as np

# Define range of HSV & set threshold
lower_blue = np.array([100, 100, 100])
upper_blue = np.array([140, 255, 255])

lower_green = np.array([50, 100, 100])
upper_green = np.array([70, 255, 255])

lower_red = np.array([-20, 100, 100])
upper_red = np.array([20, 255, 255])

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()

    # BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
    mask_green = cv2.inRange(hsv, lower_green, upper_green)
    mask_red = cv2.inRange(hsv, lower_red, upper_red)

    # masking
    res1 = cv2.bitwise_and(frame, frame, mask=mask_blue)
    res2 = cv2.bitwise_and(frame, frame, mask=mask_green)
    res3 = cv2.bitwise_and(frame, frame, mask=mask_red)

    cv2.imshow('original', frame)
    cv2.imshow('BLUE', res1)
    cv2.imshow('GREEN', res2)
    cv2.imshow('RED', res3)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:   # ESC key
        break

cv2.destroyAllWindows()