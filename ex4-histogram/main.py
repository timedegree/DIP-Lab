import numpy as np
from skimage import io, exposure, data
import matplotlib.pyplot as plt

dct4 = io.imread('./dct4.jpg')
histeq1 = io.imread('./histeq1.jpg')
coffee = data.coffee()

dct4_histogram = exposure.histogram(dct4)
histeq1_histogram = exposure.histogram(histeq1)

dct4_equalized = exposure.equalize_hist(exposure.equalize_hist(dct4))
histeq1_equalized = exposure.equalize_hist(exposure.equalize_hist(histeq1))

dct4_histogram_equalized = exposure.histogram(dct4_equalized)
histeq1_histogram_equalized = exposure.histogram(histeq1_equalized)

plt.figure(figsize=(16, 8))
plt.subplot(241)
plt.imshow(dct4, cmap='gray')
plt.title('Dct4 (Original)')
plt.subplot(242)
plt.hist(dct4.flatten(), bins=256, color='blue', density=True, edgecolor='None')
plt.title('Histogram Dct4 (Original)')
plt.subplot(243)
plt.imshow(histeq1, cmap='gray')
plt.title('Histeq1 (Original)')
plt.subplot(244)
plt.hist(histeq1.flatten(), bins=256, color='blue', density=True, edgecolor='None')
plt.title('Histogram Histeq1 (Original)')
plt.subplot(245)
plt.imshow(dct4_equalized, cmap='gray')
plt.title('Dct4 (Equalized)')
plt.subplot(246)
plt.hist(dct4_equalized.flatten(), bins=256, color='blue', density=True, edgecolor='None')
plt.title('Histogram Dct4 (Equalized)')
plt.subplot(247)
plt.imshow(histeq1_equalized, cmap='gray')
plt.title('Histeq1 (Equalized)')
plt.subplot(248)
plt.hist(histeq1_equalized.flatten(), bins=256, color='blue', density=True, edgecolor='None')
plt.title('Histogram Histeq1 (Equalized)')
plt.subplots_adjust(wspace=0.5)
plt.show()

coffee_r = coffee[:, :, 0]
coffee_g = coffee[:, :, 1]
coffee_b = coffee[:, :, 2]

coffee_r_equalized = exposure.equalize_hist(coffee_r)
coffee_g_equalized = exposure.equalize_hist(coffee_g)
coffee_b_equalized = exposure.equalize_hist(coffee_b)

coffee_equalized = np.stack((coffee_r_equalized, coffee_g_equalized, coffee_b_equalized), axis=2)

plt.figure(figsize=(16, 8))
plt.subplot(241)
plt.imshow(coffee)
plt.title('Coffee (Original)')
plt.subplot(242)
plt.hist(coffee_r.flatten(), bins=256, color='red', density=True, edgecolor='None')
plt.subplot(243)
plt.hist(coffee_g.flatten(), bins=256, color='green', density=True, edgecolor='None')
plt.title('Histogram Coffee (Original)')
plt.subplot(244)
plt.hist(coffee_b.flatten(), bins=256, color='blue', density=True, edgecolor='None')
plt.subplot(245)
plt.imshow(coffee_equalized)
plt.title('Coffee (Equalized)')
plt.subplot(246)
plt.hist(coffee_r_equalized.flatten(), bins=256, color='red', density=True, edgecolor='None')
plt.subplot(247)
plt.hist(coffee_g_equalized.flatten(), bins=256, color='green', density=True, edgecolor='None')
plt.title('Histogram Coffee (Equalized)')
plt.subplot(248)
plt.hist(coffee_b_equalized.flatten(), bins=256, color='blue', density=True, edgecolor='None')
plt.subplots_adjust(wspace=0.5)
plt.show()