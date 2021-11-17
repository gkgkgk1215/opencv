import cv2
from matplotlib import pyplot as plt

img = cv2.imread('example_images/dgist_building.jpg', cv2.IMREAD_GRAYSCALE)
threshold = 127
ret1, thr1 = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)
ret2, thr2 = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY_INV)
ret3, thr3 = cv2.threshold(img, threshold, 255, cv2.THRESH_TRUNC)
ret4, thr4 = cv2.threshold(img, threshold, 255, cv2.THRESH_TOZERO)
ret5, thr5 = cv2.threshold(img, threshold, 255, cv2.THRESH_TOZERO_INV)

# Adaptive thresholding
thr6 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
thr7 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

titles = ['original', 'BINARY', 'BINARY_INV', 'TRUNC', 'TOZERO', 'TOZERO_INV', 'ADAPTIVE_MEAN', 'ADAPTIVE_GAUSSIAN']
images = [img, thr1, thr2, thr3, thr4, thr5, thr6, thr7]

for i in range(len(titles)):
    cv2.imshow(titles[i], images[i])

# Otsu binarization
ret, thr8 = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Otsu binarization after Gaussian blurring
blur = cv2.GaussianBlur(img, (5, 5), 0)
ret, thr9 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

titles = ['original_noisy', 'Histogram', 'G-Thresholding',
          'original_noisy', 'Histogram', 'Otsu Thresholding',
          'Gaussian-filtered', 'Histogram', 'Otsu Thresholding']
images = [img, 0, thr1, img, 0, thr8, blur, 0, thr9]

for i in range(3):
    plt.subplot(3, 3, i * 3 + 1), plt.imshow(images[i * 3], 'gray')
    plt.title(titles[i * 3]), plt.xticks([]), plt.yticks([])

    plt.subplot(3, 3, i * 3 + 2), plt.hist(images[i * 3].ravel(), 256)
    plt.title(titles[i * 3 + 1]), plt.xticks([]), plt.yticks([])

    plt.subplot(3, 3, i * 3 + 3), plt.imshow(images[i * 3 + 2], 'gray')
    plt.title(titles[i * 3 + 2]), plt.xticks([]), plt.yticks([])

plt.show()
cv2.waitKey(0)
cv2.destroyAllWindows()