from skimage import io, data, exposure, color
import matplotlib.pyplot as plt
import numpy as np

img_coffee = data.coffee()
split_img_coffee = img_coffee[:,60:-60,:]

split_img_coffee_with_noise = split_img_coffee.copy()
rows,cols,dims = split_img_coffee.shape
for i in range(5000):
    x = np.random.randint(0,rows)
    y = np.random.randint(0,cols)
    split_img_coffee_with_noise[x,y,:] = 255

split_img_coffee_gamma = exposure.adjust_gamma(split_img_coffee, gamma=1.5)
while(exposure.is_low_contrast(split_img_coffee_gamma)):
    split_img_coffee_gamma = exposure.adjust_gamma(split_img_coffee_gamma, gamma=1.5)

split_img_coffee_log   = exposure.adjust_log(split_img_coffee)
while(exposure.is_low_contrast(split_img_coffee_log)):
    split_img_coffee_log = exposure.adjust_log(split_img_coffee_log)

split_img_coffee_gamma_binarization = exposure.rescale_intensity(color.rgb2gray(split_img_coffee_gamma.copy()),out_range=(0,255))
split_img_coffee_log_binarization   = exposure.rescale_intensity(color.rgb2gray(split_img_coffee_log.copy()),out_range=(0,255))

split_img_coffee_gamma_binarization[split_img_coffee_gamma_binarization <= 180] = 0
split_img_coffee_gamma_binarization[split_img_coffee_gamma_binarization > 180] = 1
split_img_coffee_log_binarization[split_img_coffee_log_binarization <= 180] = 0
split_img_coffee_log_binarization[split_img_coffee_log_binarization > 180] = 1

split_img_coffee_gamma_binarization = exposure.rescale_intensity(split_img_coffee_gamma_binarization)
split_img_coffee_log_binarization   = exposure.rescale_intensity(split_img_coffee_log_binarization)

plt.figure(figsize=(10, 8))
plt.subplot(231)
plt.imshow(split_img_coffee)
plt.title('(2) Spilt')
plt.subplot(234)
plt.imshow(split_img_coffee_with_noise)
plt.title('(2) Noise')
plt.subplot(232)
plt.imshow(split_img_coffee_gamma)
plt.title('(3) Gamma')
plt.subplot(233)
plt.imshow(split_img_coffee_log)
plt.title('(3) Log')
plt.subplot(235)
plt.imshow(split_img_coffee_gamma_binarization,cmap=plt.cm.gray)
plt.title('(4) Gamma Binarization')
plt.subplot(236)
plt.imshow(split_img_coffee_log_binarization,cmap=plt.cm.gray)
plt.title('(4) Log Binarization')
plt.show()