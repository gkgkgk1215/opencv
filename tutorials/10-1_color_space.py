import numpy as np
import cv2

I = cv2.imread("images/kittens.jpg")
cv2.imshow("Original image", I); cv2.waitKey(0)

# Convert to Grayscale
Igray = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
cv2.imshow("Gray image", Igray); cv2.waitKey(0)
#cv2.destroyAllWindows()

# Convert to HSV
Ihsv = cv2.cvtColor(I, cv2.COLOR_BGR2HSV)
# Extract and show only the Hue component
Ihue = Ihsv[:,:,0]
cv2.imshow("Hue", Ihue); cv2.waitKey(0)

# Segment the yellow color using the "hue"
# ----------------------------------------
# Lower and upper bounds for yellow
lower_yellow = np.array([20, 50, 50])
upper_yellow = np.array([40, 255, 255])
# Mask that selects pixels within the bounds
mask = cv2.inRange(Ihsv, lower_yellow, upper_yellow)
# Apply the mast to the image to keep only the desired region
Iyellow = cv2.bitwise_and(I, I, mask=mask)

# Show the image
cv2.imshow("Yellow part of the image", Iyellow)
cv2.waitKey(0)
cv2.destroyAllWindows()