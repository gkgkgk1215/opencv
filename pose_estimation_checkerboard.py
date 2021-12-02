import numpy as np
import cv2

def drawCube(img, imgpts):
    imgpts = np.int32(imgpts).reshape(-1, 2)
    cv2.drawContours(img,  [imgpts[:4]], -1, (255, 0, 0), -3)

    for i, j in zip(range(4), range(4, 8)):
        cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]), (0, 255, 0), 2)

    cv2.drawContours(img, [imgpts[4:]], -1, (0, 255, 0), 2)
    return img.copy()

row = 9
col = 13
grid_size = 136/10  # mm
mtx = np.load('camera_intrinsics/K.npy')
dist = np.load('camera_intrinsics/D.npy')

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((row * col, 3), np.float32)
objp[:, :2] = np.mgrid[0:col, 0:row].T.reshape(-1, 2)
objp = objp*grid_size

# cube vertex
L = 13.6*2   # (mm)
axis = np.float32([[0, 0, 0], [0, L, 0], [L, L, 0], [L, 0, 0],
                   [0, 0, -3*L], [0, L, -3*L], [L, L, -3*L], [L, 0, -3*L]])

# Arrays to store object points and image points from all the images.
objpoints = []  # 3d point in real world space
imgpoints = []  # 2d points in image plane.

cam = cv2.VideoCapture(0)
while True:
    # Read frame from camera
    ret, frame = cam.read()
    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (col, row), None)

    # If found, add object points, image points (after refining them)
    if ret == True:
        cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        _, rvecs, tvecs, inliers = cv2.solvePnPRansac(objp, corners, mtx, dist)
        # print ("rvecs=", rvecs.reshape(-1))
        print ("tvecs=", tvecs.reshape(-1))
        # print ("")
        imgpts, jac = cv2.projectPoints(axis, rvecs, tvecs, mtx, dist)
        frame = drawCube(frame, imgpts)

    cv2.imshow("AR", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()


