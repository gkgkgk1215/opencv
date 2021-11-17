import numpy as np
import cv2

# Load an image
img = cv2.imread("example_images/dgist.jpg", cv2.IMREAD_COLOR)

# Method1: show using OpenCV
cv2.imshow('DGIST Logo', img)   # show the image
cv2.waitKey(0)                  # wait until any key is pressed
cv2.destroyAllWindows()

print (img)
print (np.shape(img))   # print size of the image
# import pdb; pdb.set_trace()

# Method2: Show using matplotlib
from matplotlib import pyplot as plt
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img_rgb)
plt.show()