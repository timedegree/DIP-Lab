from skimage import io
import numpy as np
import matplotlib.pyplot as plt

def lpfilter(flag, rows, cols, d0, n):
    assert d0 > 0
    fliter = np.zeros((rows, cols))
    x0 = np.floor(rows / 2)
    y0 = np.floor(cols / 2)

    # ideal low pass filter
    if flag == 0:
        for i in range(rows):
            for j in range(cols):
                d = np.sqrt((i - x0) ** 2 + (j - y0) ** 2)
                if d <= d0:
                    fliter[i, j] = 1
    # Butterworth low pass filter
    elif flag == 1:
        for i in range(rows):
            for j in range(cols):
                d = np.sqrt((i - x0) ** 2 + (j - y0) ** 2)
                fliter[i, j] = 1 / (1 + (d / d0) ** (2 * n))
    # Gaussian low pass filter
    elif flag == 2:
        for i in range(rows):
            for j in range(cols):
                d = np.sqrt((i - x0) ** 2 + (j - y0) ** 2)
                fliter[i, j] = np.exp((-1) * d ** 2 / (2 * (d0 ** 2)))

    return fliter

def hpfilter(flag, rows, cols, d0, n):
    assert d0 > 0
    fliter = np.zeros((rows, cols))
    x0 = np.floor(rows / 2)
    y0 = np.floor(cols / 2)

    # ideal high pass filter
    if flag == 0:
        for i in range(rows):
            for j in range(cols):
                d = np.sqrt((i - x0) ** 2 + (j - y0) ** 2)
                if d > d0:
                    fliter[i, j] = 1
    # Butterworth high pass filter
    elif flag == 1:
        for i in range(rows):
            for j in range(cols):
                d = np.sqrt((i - x0) ** 2 + (j - y0) ** 2)
                fliter[i, j] = 1 / (1 + (d0 / (d + 1e7)) ** (2 * n))
    # Gaussian high pass filter
    elif flag == 2:
        for i in range(rows):
            for j in range(cols):
                d = np.sqrt((i - x0) ** 2 + (j - y0) ** 2)
                fliter[i, j] = 1 - np.exp((-1) * d ** 2 / (2 * (d0 ** 2)))

    return fliter


charA = io.imread('./characterA.jpg')

charA_FFT = np.fft.fft2(charA)
charA_FFT_shifted = np.fft.fftshift(charA_FFT)

ideal_lpfilter_d15 = lpfilter(0, charA.shape[0], charA.shape[1], 15, 0)
bw_lpfilter_d15_n1 = lpfilter(1, charA.shape[0], charA.shape[1], 15, 1)
bw_lpfilter_d15_n2 = lpfilter(1, charA.shape[0], charA.shape[1], 15, 2)
bw_lpfilter_d15_n5 = lpfilter(1, charA.shape[0], charA.shape[1], 15, 5)
gaussian_lpfilter_d15 = lpfilter(2, charA.shape[0], charA.shape[1], 15, 0)
ideal_lpfilter_d200 = lpfilter(0, charA.shape[0], charA.shape[1], 200, 0)
bw_lpfilter_d200_n1 = lpfilter(1, charA.shape[0], charA.shape[1], 200, 1)
bw_lpfilter_d200_n2 = lpfilter(1, charA.shape[0], charA.shape[1], 200, 2)
bw_lpfilter_d200_n5 = lpfilter(1, charA.shape[0], charA.shape[1], 200, 5)
gaussian_lpfilter_d200 = lpfilter(2, charA.shape[0], charA.shape[1], 200, 0)

charA_filtered_ideal_lp_d15 = np.fft.ifft2(np.fft.ifftshift(charA_FFT_shifted * ideal_lpfilter_d15))
charA_filtered_bw_lp_d15_n1 = np.fft.ifft2(np.fft.ifftshift(charA_FFT_shifted * bw_lpfilter_d15_n1))
charA_filtered_bw_lp_d15_n2 = np.fft.ifft2(np.fft.ifftshift(charA_FFT_shifted * bw_lpfilter_d15_n2))
charA_filtered_bw_lp_d15_n5 = np.fft.ifft2(np.fft.ifftshift(charA_FFT_shifted * bw_lpfilter_d15_n5))
charA_filtered_gaussian_lp_d15 = np.fft.ifft2(np.fft.ifftshift(charA_FFT_shifted * gaussian_lpfilter_d15))
charA_filtered_ideal_lp_d200 = np.fft.ifft2(np.fft.ifftshift(charA_FFT_shifted * ideal_lpfilter_d200))
charA_filtered_bw_lp_d200_n1 = np.fft.ifft2(np.fft.ifftshift(charA_FFT_shifted * bw_lpfilter_d200_n1))
charA_filtered_bw_lp_d200_n2 = np.fft.ifft2(np.fft.ifftshift(charA_FFT_shifted * bw_lpfilter_d200_n2))
charA_filtered_bw_lp_d200_n5 = np.fft.ifft2(np.fft.ifftshift(charA_FFT_shifted * bw_lpfilter_d200_n5))
charA_filtered_gaussian_lp_d200 = np.fft.ifft2(np.fft.ifftshift(charA_FFT_shifted * gaussian_lpfilter_d200))

plt.figure(figsize=(15, 8))
plt.subplot(251)
plt.imshow(np.abs(charA_filtered_ideal_lp_d15), cmap='gray')
plt.title('Ideal LP Filter D=15')
plt.subplot(252)
plt.imshow(np.abs(charA_filtered_bw_lp_d15_n1), cmap='gray')
plt.title('Butterworth LP Filter D=15 N=1')
plt.subplot(253)
plt.imshow(np.abs(charA_filtered_bw_lp_d15_n2), cmap='gray')
plt.title('Butterworth LP Filter D=15 N=2')
plt.subplot(254)
plt.imshow(np.abs(charA_filtered_bw_lp_d15_n5), cmap='gray')
plt.title('Butterworth LP Filter D=15 N=5')
plt.subplot(255)
plt.imshow(np.abs(charA_filtered_gaussian_lp_d15), cmap='gray')
plt.title('Gaussian LP Filter D=15')
plt.subplot(256)
plt.imshow(np.abs(charA_filtered_ideal_lp_d200), cmap='gray')
plt.title('Ideal LP Filter D=200')
plt.subplot(257)
plt.imshow(np.abs(charA_filtered_bw_lp_d200_n1), cmap='gray')
plt.title('Butterworth LP Filter D=200 N=1')
plt.subplot(258)
plt.imshow(np.abs(charA_filtered_bw_lp_d200_n2), cmap='gray')
plt.title('Butterworth LP Filter D=200 N=2')
plt.subplot(259)
plt.imshow(np.abs(charA_filtered_bw_lp_d200_n5), cmap='gray')
plt.title('Butterworth LP Filter D=200 N=5')
plt.subplot(2, 5, 10)
plt.imshow(np.abs(charA_filtered_gaussian_lp_d200), cmap='gray')
plt.title('Gaussian LP Filter D=200')
plt.tight_layout()
plt.show()


moon = io.imread('./moon.jpg')

moon_FFT = np.fft.fft2(moon)
moon_FFT_shifted = np.fft.fftshift(moon_FFT)

ideal_hpfilter_d15 = hpfilter(0, moon.shape[0], moon.shape[1], 15, 0)
bw_hpfilter_d15_n1 = hpfilter(1, moon.shape[0], moon.shape[1], 15, 1)
bw_hpfilter_d15_n2 = hpfilter(1, moon.shape[0], moon.shape[1], 15, 2)
bw_hpfilter_d15_n5 = hpfilter(1, moon.shape[0], moon.shape[1], 15, 5)
gaussian_hpfilter_d15 = hpfilter(2, moon.shape[0], moon.shape[1], 15, 0)
ideal_hpfilter_d200 = hpfilter(0, moon.shape[0], moon.shape[1], 200, 0)
bw_hpfilter_d200_n1 = hpfilter(1, moon.shape[0], moon.shape[1], 200, 1)
bw_hpfilter_d200_n2 = hpfilter(1, moon.shape[0], moon.shape[1], 200, 2)
bw_hpfilter_d200_n5 = hpfilter(1, moon.shape[0], moon.shape[1], 200, 5)
gaussian_hpfilter_d200 = hpfilter(2, moon.shape[0], moon.shape[1], 200, 0)

moon_filtered_ideal_hp_d15 = np.fft.ifft2(np.fft.ifftshift(moon_FFT_shifted * ideal_hpfilter_d15))
moon_filtered_bw_hp_d15_n1 = np.fft.ifft2(np.fft.ifftshift(moon_FFT_shifted * bw_hpfilter_d15_n1))
moon_filtered_bw_hp_d15_n2 = np.fft.ifft2(np.fft.ifftshift(moon_FFT_shifted * bw_hpfilter_d15_n2))
moon_filtered_bw_hp_d15_n5 = np.fft.ifft2(np.fft.ifftshift(moon_FFT_shifted * bw_hpfilter_d15_n5))
moon_filtered_gaussian_hp_d15 = np.fft.ifft2(np.fft.ifftshift(moon_FFT_shifted * gaussian_hpfilter_d15))
moon_filtered_ideal_hp_d200 = np.fft.ifft2(np.fft.ifftshift(moon_FFT_shifted * ideal_hpfilter_d200))
moon_filtered_bw_hp_d200_n1 = np.fft.ifft2(np.fft.ifftshift(moon_FFT_shifted * bw_hpfilter_d200_n1))
moon_filtered_bw_hp_d200_n2 = np.fft.ifft2(np.fft.ifftshift(moon_FFT_shifted * bw_hpfilter_d200_n2))
moon_filtered_bw_hp_d200_n5 = np.fft.ifft2(np.fft.ifftshift(moon_FFT_shifted * bw_hpfilter_d200_n5))
moon_filtered_gaussian_hp_d200 = np.fft.ifft2(np.fft.ifftshift(moon_FFT_shifted * gaussian_hpfilter_d200))

plt.figure(figsize=(15, 8))
plt.subplot(251)
plt.imshow(np.abs(moon_filtered_ideal_hp_d15), cmap='gray')
plt.title('Ideal HP Filter D=15')
plt.subplot(252)
plt.imshow(np.abs(moon_filtered_bw_hp_d15_n1), cmap='gray')
plt.title('Butterworth HP Filter D=15 N=1')
plt.subplot(253)
plt.imshow(np.abs(moon_filtered_bw_hp_d15_n2), cmap='gray')
plt.title('Butterworth HP Filter D=15 N=2')
plt.subplot(254)
plt.imshow(np.abs(moon_filtered_bw_hp_d15_n5), cmap='gray')
plt.title('Butterworth HP Filter D=15 N=5')
plt.subplot(255)
plt.imshow(np.abs(moon_filtered_gaussian_hp_d15), cmap='gray')
plt.title('Gaussian HP Filter D=15')
plt.subplot(256)
plt.imshow(np.abs(moon_filtered_ideal_hp_d200), cmap='gray')
plt.title('Ideal HP Filter D=200')
plt.subplot(257)
plt.imshow(np.abs(moon_filtered_bw_hp_d200_n1), cmap='gray')
plt.title('Butterworth HP Filter D=200 N=1')
plt.subplot(258)
plt.imshow(np.abs(moon_filtered_bw_hp_d200_n2), cmap='gray')
plt.title('Butterworth HP Filter D=200 N=2')
plt.subplot(259)
plt.imshow(np.abs(moon_filtered_bw_hp_d200_n5), cmap='gray')
plt.title('Butterworth HP Filter D=200 N=5')
plt.subplot(2, 5, 10)
plt.imshow(np.abs(moon_filtered_gaussian_hp_d200), cmap='gray')
plt.title('Gaussian HP Filter D=200')
plt.tight_layout()
plt.show()