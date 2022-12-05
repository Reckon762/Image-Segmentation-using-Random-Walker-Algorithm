import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
from skimage.segmentation import random_walker
from skimage import img_as_float
from skimage.restoration import estimate_sigma

# Image is read using openCV
img = img_as_float(cv.imread("Dataset\\2.png"))

# Transforming the image to 2D
img = img[0:128,0:128,1]

# plt.imshow(img, cmap = 'gray')
# print(img.shape)

# finding sigma for calculating weight of edge
sigma = np.mean(estimate_sigma(img, channel_axis=True))

# plt.hist(img.flat, bins=100, range=(0, 1))

# The intensity of the binary image lies in range (0, 1).
# We choose the hottest and the coldest pixels as markers.
markers = np.zeros(img.shape, dtype=np.uint)
markers[img > 0.7] = 2
markers[img <0.2] = 1

# Applying random walker algorithm on image using suitable markers
labels = random_walker(img, markers, beta=25, mode='bf')

# Obtaing segments
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 3.2),
                                    sharex=True, sharey=True)
ax1.imshow(img, cmap='gray')
ax1.axis('off')
ax1.set_title('Input Image')
ax2.imshow(labels, cmap='gray')
ax2.axis('off')
ax2.set_title('Segmentation')

fig.tight_layout()
plt.show()