import cv2

def OnMouse(x):
    pass

img1 = cv2.imread('example_images/scene_960_540.jpg')
img2 = cv2.imread('example_images/animation_960_540.jpg')

cv2.imshow('img1', img1)
cv2.waitKey(0)
cv2.imshow('img2', img2)
cv2.waitKey(0)
cv2.imshow('img1+img2', img1+img2)
cv2.waitKey(0)
cv2.imshow('img1+img2', cv2.add(img1, img2))
cv2.waitKey(0)

cv2.namedWindow('ImgPane')
cv2.createTrackbar('MIXING', 'ImgPane', 0, 100, OnMouse)
mix = cv2.getTrackbarPos('MIXING', 'ImgPane')

while True:
    # image blending
    # g(x) = (1-alpha)*f1(x) + alpha*f2(x)
    img = cv2.addWeighted(img1, float(100 - mix) / 100, img2, float(mix) / 100, 0)
    cv2.imshow('ImgPane', img)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

    mix = cv2.getTrackbarPos('MIXING', 'ImgPane')
cv2.destroyAllWindows()