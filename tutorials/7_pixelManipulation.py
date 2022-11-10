import cv2

img = cv2.imread('example_images/dgist_building.jpg')
print('shape:', img.shape)
print('size:', img.size, '(byte)')
print('data type:', img.dtype)

px = img[200, 300]
print(px)

# B = img.item(200, 300, 0)
# G = img.item(200, 300, 1)
# R = img.item(200, 300, 2)
# BGR = [B, G, R]
# print(BGR)
# img.itemset((200,300,0), 100)   # change blue=100 at pixel (200,300)

cv2.imshow('original', img)

subimg = img[50:400, 100:400]  # img cutting
cv2.imshow('cutting', subimg)

subimg_gray = cv2.cvtColor(subimg, cv2.COLOR_BGR2LAB)
img[50:400, 100:400] = subimg_gray
cv2.imshow('modified', img)

cv2.waitKey(0)
cv2.destroyAllWindows()