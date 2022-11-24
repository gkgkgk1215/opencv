import numpy as np
import cv2

I = cv2.imread('images/shapes.png', 0)

# Structuring element
se = cv2.getStructuringElement(cv2.MORPH_RECT, (9,9))
# cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
# cv2.getStructuringElement(cv2.MORPH_CROSS, (5,5))

# Basic morphology
Ierosion  = cv2.erode(I, se, iterations=1)
Idilation = cv2.dilate(I, se, iterations=1)
Iopening  = cv2.morphologyEx(I, cv2.MORPH_OPEN, se)
Iclosing  = cv2.morphologyEx(I, cv2.MORPH_CLOSE, se)

cv2.imshow('original', I)
cv2.imshow('eroded', Ierosion)
cv2.imshow('dilated', Idilation)
cv2.imshow('opened', Iopening)
cv2.imshow('closed', Iclosing)
cv2.waitKey(0)


# Difference between dilation and erosion
se2 = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
gradient = cv2.morphologyEx(I, cv2.MORPH_GRADIENT, se2)
cv2.imshow('gradient', gradient)
cv2.waitKey(0)
cv2.destroyAllWindows()
