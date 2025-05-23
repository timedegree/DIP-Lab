from skimage import io,color
import skimage.morphology as sm
import matplotlib.pyplot as plt
import numpy as np

img1 = io.imread('形态学1.bmp')
img2 = color.rgb2gray(io.imread('形态学2.bmp'))

img2_opening = sm.opening(img2, sm.disk(3))
img2_closing = sm.closing(img2, sm.disk(3))

plt.figure(figsize=(12, 8))
plt.subplot(131)
plt.imshow(img2, cmap='gray')
plt.title('Original Image')
plt.subplot(132)
plt.imshow(img2_opening, cmap='gray')
plt.title('Opening (D = 3)')
plt.subplot(133)
plt.imshow(img2_closing, cmap='gray')
plt.title('Closing (D = 3)')
plt.show()

img1_remove_small = sm.erosion(sm.closing(img1, sm.disk(4)), sm.disk(3))
img1_convex = sm.convex_hull_image(img1_remove_small)

plt.figure(figsize=(12, 8))
plt.subplot(131)
plt.imshow(img1, cmap='gray')
plt.title('Original Image')
plt.subplot(132)
plt.imshow(img1_remove_small, cmap='gray')
plt.title('Remove Small Objects')
plt.subplot(133)
plt.imshow(img1_convex, cmap='gray')
plt.title('Convex Hull')
plt.show()