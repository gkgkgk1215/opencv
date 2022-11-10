import numpy as np
import cv2

# define names of each possible ArUco tag OpenCV supports
ARUCO_DICT = {
    "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
    "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
    "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
    "DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
    "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
    "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
    "DICT_5X5_250": cv2.aruco.DICT_5X5_250,
    "DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
    "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
    "DICT_6X6_100": cv2.aruco.DICT_6X6_100,
    "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
    "DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
    "DICT_7X7_50": cv2.aruco.DICT_7X7_50,
    "DICT_7X7_100": cv2.aruco.DICT_7X7_100,
    "DICT_7X7_250": cv2.aruco.DICT_7X7_250,
    "DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
    "DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
    "DICT_APRILTAG_16h5": cv2.aruco.DICT_APRILTAG_16h5,
    "DICT_APRILTAG_25h9": cv2.aruco.DICT_APRILTAG_25h9,
    "DICT_APRILTAG_36h10": cv2.aruco.DICT_APRILTAG_36h10,
    "DICT_APRILTAG_36h11": cv2.aruco.DICT_APRILTAG_36h11
}

# load the input image from disk and resize it
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

