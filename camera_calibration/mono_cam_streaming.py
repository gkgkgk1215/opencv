import cv2
from ImgUtils import ImgUtils

cam = cv2.VideoCapture(0)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 960)

while cv2.waitKey(1):
    ret, frame = cam.read()
    if ret:
        # compared = ImgUtils.compare_rectified_img(frameL, frameR, scale=1.0)
        cv2.imshow("", frame)

cam.release()
# camR.release()
cv2.destroyAllWindows()