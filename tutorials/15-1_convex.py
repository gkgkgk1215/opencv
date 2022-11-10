import cv2

src = cv2.imread("example_images/convex_test.jpg")
dst = src.copy()

gray = cv2.cvtColor(src, cv2.COLOR_RGB2GRAY)
threshold = 150
ret, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)
contours, hierarchy = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)

for cnt in contours:
    cv2.drawContours(dst, [cnt], 0, (255, 0, 0), 2)
    hull = cv2.convexHull(cnt, clockwise=True)
    cv2.drawContours(dst, [hull], 0, (0, 0, 255), 2)

cv2.imshow("dst", dst)
cv2.waitKey(0)
cv2.destroyAllWindows()