import cv2
from ImgUtils import ImgUtils

# Left camera
camL = cv2.VideoCapture(2)
camL.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
camL.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Right camera
camR = cv2.VideoCapture(0)
camR.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
camR.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
while cv2.waitKey(1):
    retL, frameL = camL.read()
    retR, frameR = camR.read()
    if retL and retR:
        compared = ImgUtils.compare_rectified_img(frameL, frameR, scale=1.0)
        cv2.imshow("", compared)

camL.release()
# camR.release()
cv2.destroyAllWindows()