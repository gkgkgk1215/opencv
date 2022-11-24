import cv2
import numpy as np

I = cv2.imread('images/inside.jpg')
Igray = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
Iedges = cv2.Canny(Igray, 150, 200)

# Classic Hough transform
distResolution = 1
angleResolution = 1*np.pi/180
minVotes = 150
lines = cv2.HoughLines(Iedges, distResolution, angleResolution, minVotes)
for line in lines:
    for rho,theta in line:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))
        cv2.line(I,(x1,y1),(x2,y2),(0,0,255),2)
    
cv2.imshow('edges', Iedges)
cv2.imshow('result', I)
cv2.waitKey(0); cv2.destroyAllWindows()

# Probabilistic Hough transform
I = cv2.imread('images/inside.jpg')
distResolution = 1
angleResolution = 1*np.pi/180
minVotes = 100
minLineLength = 200
maxLineGap = 20
lines = cv2.HoughLinesP(Iedges, distResolution, angleResolution, minVotes, minLineLength, maxLineGap)
for line in lines:
    for x1,y1,x2,y2 in line:
        cv2.line(I,(x1,y1),(x2,y2),(0,255,0),2)

cv2.imshow('prob result', I)
cv2.waitKey(0); cv2.destroyAllWindows()

# -------------------------------------
# Hough transform applied to circles
# -------------------------------------

I = cv2.imread('images/coins.jpg')
Igray = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
Igray = cv2.medianBlur(Igray, 5)

minDist = 30 # Min distance btw center of detected HoughLines
THCanny = 150 # High threshold of Canny edge detector
minVotes = 100
circles = cv2.HoughCircles(Igray, cv2.HOUGH_GRADIENT, 1, minDist, param1=THCanny,
                           param2=minVotes, minRadius=10, maxRadius=200)

circles = np.uint16(np.around(circles))
for i in circles[0,:]:
    # Draw the outer circle
    cv2.circle(I,(i[0],i[1]),i[2],(0,255,0),2)
    # Draw the center of the circle
    cv2.circle(I,(i[0],i[1]),2,(0,0,255),3)

cv2.imshow('detected circles', I)
cv2.waitKey(0)
cv2.destroyAllWindows()
