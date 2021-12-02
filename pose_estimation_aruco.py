import numpy as np
import cv2
from aruco_generator import ARUCO_DICT

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

# Load aruco dictionary
type = "DICT_6X6_50"
arucoDict = cv2.aruco.Dictionary_get(ARUCO_DICT[type])

# Grab the ArUCo parameters
arucoParams = cv2.aruco.DetectorParameters_create()
cam = cv2.VideoCapture(0)
while True:
    # Read frame from camera
    ret, frame = cam.read()
    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, rejected = cv2.aruco.detectMarkers(image=gray, dictionary=arucoDict, parameters=arucoParams)

    # import pdb; pdb.set_trace()
    if corners:
        # Display
        for corner, id in zip(corners, ids):
            corner = corner[0]
            for pt in corner:
                cv2.circle(img=frame, center=pt.astype(int), radius=5, color=(0, 255, 0), thickness=-1)
            center = (corner[0]+corner[2])/2
            cv2.putText(img=frame, text=str(id), org=center.astype(int)+np.array([0, 90]), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 0, 0), thickness=2, lineType=cv2.LINE_AA)
    cv2.imshow("Detected result", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

# K = np.load('camera_intrinsics/K.npy')
# D = np.load('camera_intrinsics/D.npy')
# rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners=corners[0].astype(np.float64), markerLength=0.10, cameraMatrix=K, distCoeffs=D)