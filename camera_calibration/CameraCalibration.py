import cv2
import numpy as np
from OV9750 import OV9750

# grid information
row = 6
col = 9
grid_size = 12.5  # mm

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((row * col, 3), np.float32)
objp[:, :2] = np.mgrid[0:col, 0:row].T.reshape(-1, 2)
objp = objp*grid_size

# Arrays to store object points and image points from all the images.
objpoints = []  # 3d point in real world space
imgpoints = []  # 2d points in image plane.

# cam = cv2.VideoCapture(0)
cam = OV9750()

while True:
    # Read frame from camera
    # ret, frame = cam.read()
    img_L, img_R = cam.get_frame()
    frame = img_R
    cv2.imshow("camera frame", frame)

    # Key input
    key = cv2.waitKey(1) & 0xFF
    if key == ord('\r'):    # ENTER
        # Find the chess board corners
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (col, row), None)

        # If found, add object points, image points (after refining them)
        if ret == True:
            corners_ref = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

            # append points
            imgpoints.append(corners_ref)
            objpoints.append(objp)

            # Draw and display the corners
            img = cv2.drawChessboardCorners(frame, (col, row), corners_ref, ret)
            print (str(len(imgpoints)), " corners are collected.")
            cv2.imshow('camera frame', img)
            cv2.waitKey(1)
    elif key == ord('q'):
        break

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
print("mtx: ", mtx)
print("dist: ", dist)
# print("rvecs: ", rvecs)
# print("tvecs: ", tvecs)
np.save("camera_intrinsics/K", mtx)
np.save("camera_intrinsics/D", dist)

# # Get new camera marix
# h, w = img.shape[:2]
# newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))   # if alpha=1, all pixels are maintained
# print(newcameramtx)
#
# # Rectify image
# # option 1)
# dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
# x, y, w, h = roi
# dst = dst[y:y + h, x:x + w]
# cv2.imwrite('calibresult1.png', dst)
#
# # option 2)
# mapx, mapy = cv2.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (w, h), 5)
# dst = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)
# x, y, w, h = roi
# dst = dst[y:y + h, x:x + w]
# cv2.imwrite('calibresult2.png', dst)