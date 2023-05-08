#!/bin/python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread
from scipy.signal import medfilt2d, gaussian
from numpy.fft import fft2, ifft2


def gaussian_kernel(k: int = 3):
	h = gaussian(k, k / 3).reshape(k, 1)
	h = np.dot(h, h.transpose())
	h /= np.sum(h)
	return h

def wiener(img: np.ndarray, kernel: np.ndarray, SNR: float):
	dummy = np.copy(img)
	dummy = fft2(dummy)
	kernel = fft2(kernel, s = img.shape)
	kernel = np.conj(kernel) / (np.abs(kernel) ** 2 + 1 / SNR)
	dummy = dummy * kernel
	dummy = np.abs(ifft2(dummy))
	return dummy


def get_noisy_image(img: np.ndarray, std: float) -> np.ndarray:
	h = img.shape[0]
	w = img.shape[1]
	eps = np.random.normal(0, std, size=h*w)
	eps = np.reshape(eps, (h, w))
	res = img.copy() + eps
	np.clip(res, 0, 255, out=res)
	return res


def get_mse(img1: np.ndarray, img2: np.ndarray) -> float:
	h = img1.shape[0]
	w = img1.shape[1]
	tmp = (img1 - img2)**2
	return np.sum(tmp) / (h * w)

def rgb2gray(rgb: np.ndarray):
	return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

def optimize(file: str, std: float, k_params: tuple, SNR_params: tuple):
	img = rgb2gray(imread(file))
	n_img = get_noisy_image(img, std=std)
	k = k_min = k_params[0]
	k_end = k_params[1]
	k_step = k_params[2]
	SNR_min = SNR_params[0]
	SNR_end = SNR_params[1]
	SNR_step = SNR_params[2]
	print("optimizing for std:", std)
	mse_min = 10000000
	
	while k < k_end:
		kernel = gaussian_kernel(k)
		SNR = SNR_params[0]
		while SNR < SNR_end:
			f_img = wiener(img=n_img, kernel=kernel, SNR=SNR)
			mse = get_mse(img, f_img)
			if mse < mse_min:
				mse_min = mse
				k_min = k
				SNR_min = SNR
			SNR += SNR_step
		k += k_step
	print("best kernel size:", k_min)
	print("best SNR:", SNR_min)
	print("MSE:", mse_min)



def test(file: str, std: float, k: int, SNR: float, m: int = 3):
	img = rgb2gray(imread(file))
	print("testing " + file)
	n_img = get_noisy_image(img, std=std)
	kernel = gaussian_kernel(k)
	f_img1 = wiener(img=n_img, kernel=kernel, SNR=SNR)
	np.clip(f_img1, 0, 255, out=f_img1)
	f_img2 = medfilt2d(input=n_img, kernel_size=m)
	_, ax = plt.subplots(2, 2, figsize=(10, 10))

	mse1 = get_mse(img, n_img)
	mse2 = get_mse(img, f_img1)
	mse3 = get_mse(img, f_img2)

	print("MSE:", round(mse1, 1))
	print("MSE Wiener:", round(mse2, 1))
	print("MSE median:", round(mse3, 1))
	ax[0][0].imshow(img, cmap = 'gray')
	ax[0][0].title.set_text("исходное")

	ax[0][1].imshow(n_img, cmap = 'gray')
	ax[0][1].title.set_text("зашумленное: std =" + str(std))

	ax[1][0].imshow(f_img1, cmap = 'gray')
	ax[1][0].title.set_text("Винер: k = " + str(k) + ", SNR ~" + str(SNR))

	ax[1][1].imshow(f_img2, cmap = 'gray')
	ax[1][1].title.set_text("медианный: m =" + str(m))
	plt.show()

#test('cat.jpg', std=200, k=18, SNR=2.3, m=5)
test('moon.jpg', std=150, k=12, SNR=3.35, m=5)
k_search = (7, 18, 1)
SNR_search = (2.0, 6.0, 0.05)
#optimize('moon.jpg', std=150, k_params=k_search, SNR_params=SNR_search)