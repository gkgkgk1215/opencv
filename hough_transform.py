import numpy as np
import cv2

"""
Searching lines
"""
img = cv2.imread("example_images/checkerboard.png")
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_edges = cv2.Canny(img_gray, 150, 200, apertureSize=3)

cv2.imshow("original", img)
cv2.waitKey(0)
cv2.imshow("edges", img_edges)
cv2.waitKey(0)

threshold = 200
lines = cv2.HoughLines(img_edges, rho=1, theta=1*np.pi/180, threshold=threshold)

for line in lines:
    r, theta = line[0]
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a*r
    y0 = b*r
    x1 = int(x0 + 1000*(-b))
    y1 = int(y0 + 1000*a)
    x2 = int(x0 - 1000*(-b))
    y2 = int(y0 - 1000 * a)
    cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 1)

cv2.imshow("result", img)
cv2.waitKey(0)
cv2.destroyAllWindows()


# """
# Searching circles
# """
# img1 = cv2.imread("example_images/coins.jpg")
# img2 = img1.copy()
#
# img2 = cv2.GaussianBlur(img2, (3,3), 0)
# img_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
#
# circles = cv2.HoughCircles(img_gray, cv2.HOUGH_GRADIENT, dp=1, minDist=10, param1=60, param2=50, minRadius=0, maxRadius=0)
#
# if circles is not None:
#     circles = np.uint16(np.around(circles))
#     for i in circles[0, :]:
#         cv2.circle(img1, (i[0], i[1]), i[2], (255, 255, 0), 5 )
#
#         cv2.imshow("Hough Circles", img1)
#         cv2.waitKey(0)
#         cv2.destroyAllWindows()
# else:
#     print ("Could not find circle")
