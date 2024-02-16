#!/usr/bin/env python3

def convolve_grayscale_same(images, kernel):
    m, h, w = images.shape
    kh, kw = kernel.shape

    pad_h = kh // 2
    pad_w = kw // 2

    images_padded = np.pad(images, ((0, 0). (pad_h, pad_h), (pad_w, pad_w)), mode='constant')

    convolved_images = np.zeros_like(images)

    for i in range(m):
        for j in range(h):
            for k in range(w):
                region = images_padded[i, j:j+kh, k:k+kw]
                convolved_value = np.sum(region * kernel)
                convolved_images[i, j, k] = convolved_value
    return convolved_images
