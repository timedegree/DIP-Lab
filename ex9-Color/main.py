from skimage import io, color, data, filters
import matplotlib.pyplot as plt
import numpy as np

img = data.chelsea()

img_rgb2gray = color.rgb2gray(img)
img_rgb2hsv = color.rgb2hsv(img)
img_hsv2rgb = color.hsv2rgb(img_rgb2hsv)
img_gray2rgb = color.gray2rgb(img_rgb2gray)

plt.figure(figsize=(16, 4))
plt.subplot(1, 4, 1)
plt.imshow(img_rgb2gray, cmap='gray')
plt.title('RGB to Gray')
plt.subplot(1, 4, 2)
plt.imshow(img_rgb2hsv)
plt.title('RGB to HSV')
plt.subplot(1, 4, 3)
plt.imshow(img_hsv2rgb)
plt.title('HSV to RGB')
plt.subplot(1, 4, 4)
plt.imshow(img_gray2rgb)
plt.title('Gray to RGB')
plt.show()

img_gray = img_rgb2gray

plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.imshow(img_gray, cmap='gray')
plt.title('Gray Image')
plt.subplot(1, 2, 2)
plt.imshow(img_gray, cmap="prism")
plt.title('Pseudocolor Image (Prism Colormap)')
plt.show()

img_noisy = img_gray + 0.1 * np.random.randn(*img_gray.shape)
img_gassian = filters.gaussian(img_noisy, sigma=1)

plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.imshow(img_noisy, cmap='hsv')
plt.title('Noisy Image (HSV)')
plt.subplot(1, 2, 2)
plt.imshow(img_gassian, cmap='hsv')
plt.title('Gaussian Filtered Image (HSV)')
plt.show()