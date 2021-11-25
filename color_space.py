import cv2
img = cv2.imread("example_images/kitten.jpg")

cv2.imshow("original image", img)
cv2.waitKey(0)

# Convert to Grayscale
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow("Gray image", img_gray)
cv2.waitKey(0)

# Convert to HSV
img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# Extract the Hue/Saturation/Value channel and show only hue component
hue = img_hsv[:, :, 0]
saturation = img_hsv[:, :, 1]
value = img_hsv[:, :, 2]
cv2.imshow("Hue", hue)
cv2.waitKey(0)

