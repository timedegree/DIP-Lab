from skimage import io, data, transform
import matplotlib.pyplot as plt
import numpy as np

img_coffee = data.coffee()

img_coffee_half = transform.rescale(img_coffee, 0.5, channel_axis=2)
img_coffee_double = transform.rescale(img_coffee, 2, channel_axis=2)

img_coffee_half_rotate60 = transform.rotate(img_coffee_half, 60)
img_coffee_half_rotate90 = transform.rotate(img_coffee_half, 90)
img_coffee_half_rotate180 = transform.rotate(img_coffee_half, 180)


plt.figure(figsize=(10, 6))
plt.subplot(231)
plt.imshow(img_coffee)
plt.title('Original')
plt.subplot(232)
plt.imshow(img_coffee_half)
plt.title('Half')
plt.subplot(233)
plt.imshow(img_coffee_double)
plt.title('Double')
plt.subplot(234)
plt.imshow(img_coffee_half_rotate60)
plt.title('Rotate 60')
plt.subplot(235)
plt.imshow(img_coffee_half_rotate90)
plt.title('Rotate 90')
plt.subplot(236)
plt.imshow(img_coffee_half_rotate180)
plt.title('Rotate 180')
plt.show()

rows, cols, dim = img_coffee_double.shape
pyramid = tuple(transform.pyramid_gaussian(img_coffee_double, downscale=2, channel_axis=2))

sum_rows = sum(p.shape[0] for p in pyramid[1:])
composite_image = np.zeros((max(rows,sum_rows), cols + cols // 2, 3), dtype=np.double)
composite_image[:rows, :cols, :] = pyramid[0]

i_row = 0
for p in pyramid[1:]:
    n_rows, n_cols = p.shape[:2]
    composite_image[i_row:i_row + n_rows, cols:cols + n_cols] = p
    i_row += n_rows

plt.figure(figsize=(10, 6))
plt.imshow(composite_image)
plt.title('Pyramid')
plt.show()
