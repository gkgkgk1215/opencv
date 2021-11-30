import numpy as np
import cv2
from aruco_generator import ARUCO_DICT

# load the input image from disk and resize it
print("[INFO] loading image...")
image = cv2.imread("robot_base.png")

# Load aruco dictionary
type = "DICT_6X6_50"
arucoDict = cv2.aruco.Dictionary_get(ARUCO_DICT[type])

# Grab the ArUCo parameters and detect the markers
arucoParams = cv2.aruco.DetectorParameters_create()
corners, ids, rejected = cv2.aruco.detectMarkers(image=image, dictionary=arucoDict, parameters=arucoParams)

# Display
for corner, id in zip(corners, ids):
    corner = corner[0]
    for pt in corner:
        cv2.circle(img=image, center=pt.astype(int), radius=5, color=(0, 255, 0), thickness=-1)
    center = (corner[0]+corner[2])/2
    cv2.putText(img=image, text=str(id), org=center.astype(int)+np.array([0, 90]), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 0, 0), thickness=2, lineType=cv2.LINE_AA)

cv2.imshow("Detected result", image)
cv2.waitKey(0)

rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(np.array(allCorners).astype(np.float64), 0.10, K, distCoeffs=None)

