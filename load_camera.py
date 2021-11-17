import numpy as np
import cv2

cam = cv2.VideoCapture(0)   # camera instance that will capture the video

while (cam.isOpened()):
    retval, frame = cam.read()      # retval returns true when there is a valid frame from the camera
    # frame = cv2.flip(frame, 1)    # flip the image

    if retval == True:
        cv2.imshow("My camera", frame)
        if cv2.waitKey(30) & 0xFF == ord('q'):  # Exit when 'q' is pressed
            break
    else:
        break