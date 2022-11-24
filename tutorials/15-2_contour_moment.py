import cv2

src = cv2.imread('example_images/convex_test.jpg')
dst = src.copy()


gray = cv2.cvtColor(src, cv2.COLOR_RGB2GRAY)
threshold = 150
ret, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)
contours, hierarchy = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)

cnt = contours[0]
mmt = cv2.moments(cnt)
for key, val in mmt.items():
    print('%s:\t%.5f' % (key, val))

# center of moment
cx = int(mmt['m10'] / mmt['m00'])
cy = int(mmt['m01'] / mmt['m00'])
print(cx, cy)

# area & perimeter
area = cv2.contourArea(cnt)
perimeter = cv2.arcLength(cnt, True)
print('Area =', area)
print('Perimeter = ', perimeter)

for cnt in contours:
    epsilon = 0.01 * cv2.arcLength(cnt, True)
    approx_PD = cv2.approxPolyDP(cnt, epsilon, True)
    cv2.drawContours(src, [approx_PD], 0, (255, 255, 0), 2)
    cv2.imshow('contour', src)

    check = cv2.isContourConvex(cnt)
    if not check:
        hull = cv2.convexHull(cnt)
        cv2.drawContours(dst, [hull], 0, (255, 255, 0), 2)
    cv2.imshow('convex_hull', dst)

cv2.waitKey(0)
cv2.destroyAllWindows()