import numpy as np
import cv2

cam = cv2.VideoCapture(0)   # camera instance that will capture the video
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while (cam.isOpened()):
    ret, frame = cam.read()      # retval returns true when there is a valid frame from the camera
    # frame = cv2.flip(frame, 1)    # flip the image

    if ret == True:
        cv2.imshow("My camera", frame)

        k = cv2.waitKey(10) & 0xFF
        if k == ord('q'):  # Exit when 'q' is pressed
            break
    else:
        print ("cam reading error")

cam.release()
cv2.destroyAllWindows()
