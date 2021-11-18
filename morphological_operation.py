import cv2

img = cv2.imread("example_images/brain_noise.jpeg")

# Structuring element
se = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))  # also called kernel

# Basic morphology
img_erosion = cv2.erode(img, se, iterations=1)
img_dilation = cv2.dilate(img, se, iterations=1)
img_opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, se)
img_closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, se)

cv2.imshow("Original", img)
cv2.waitKey(0)
cv2.imshow("Eroded", img_erosion)
cv2.waitKey(0)
cv2.imshow("Dilated", img_dilation)
cv2.waitKey(0)
cv2.imshow("Opened", img_opening)
cv2.waitKey(0)
cv2.imshow("Closed", img_closing)
cv2.waitKey(0)
