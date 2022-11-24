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


def drawCube(img, imgpts):
    imgpts = np.int32(imgpts).reshape(-1, 2)
    cv2.drawContours(img, [imgpts[:4]], -1, (255, 0, 0), -3)

    for i, j in zip(range(4), range(4, 8)):
        cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]), (0, 255, 0), 2)

    cv2.drawContours(img, [imgpts[4:]], -1, (0, 255, 0), 2)
    return img.copy()

row = 9
col = 13
markerSize = 29.0  # mm
K = np.load('../../camera_calibration/camera_intrinsics/K.npy')
D = np.load('../../camera_calibration/camera_intrinsics/D.npy')

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.float32([[0, 0],[markerSize, 0],[markerSize, markerSize],[0, markerSize]])

# cube vertex
L = markerSize
axis = np.float32([[-L/2, -L/2, 0], [-L/2, L/2, 0], [L/2, L/2, 0], [L/2, -L/2, 0],
                   [-L/2, -L/2, -L], [-L/2, L/2, -L], [L/2, L/2, -L], [L/2, -L/2, -L]])

# Load aruco dictionary
type = "DICT_6X6_50"
arucoDict = cv2.aruco.Dictionary_get(ARUCO_DICT[type])

# Grab camera capture, ArUCo parameters, and camera intrinsics
cam = cv2.VideoCapture(0)
arucoParams = cv2.aruco.DetectorParameters_create()
while True:
    # Read frame from camera
    ret, frame = cam.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, rejected = cv2.aruco.detectMarkers(image=gray, dictionary=arucoDict, parameters=arucoParams)
    if corners:
        for corner, id in zip(corners, ids):
            # draw corners, ids
            center = (corner[0][0] + corner[0][2]) / 2
            cv2.putText(img=frame, text=str(id), org=center.astype(int) + np.array([0, 90]),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 0, 0), thickness=2,
                        lineType=cv2.LINE_AA)
            # detect pose
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners=corner.astype(np.float64), markerLength=L,
                                                                  cameraMatrix=K, distCoeffs=D)
            # draw axis
            (rvecs - tvecs).any()  # get rid of that nasty numpy value array error
            cv2.aruco.drawDetectedMarkers(frame, corners)  # Draw A square around the markers
            cv2.aruco.drawAxis(frame, K, D, rvecs, tvecs, 30)  # Draw axis

    cv2.imshow("Detected result", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
