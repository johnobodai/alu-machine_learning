#!/usr/bin/env python3
import numpy as np

"""
Same Padding on grayscale images

"""

def convolve_grayscale_same(images, kernel):
    """
    Perform a convolution with 'same' padding on grayscale images.

    Args:
        images (array): Input grayscale images.
        kernel (array): The convolution kernel.

    Returns:
        array: The convolved images with 'same' padding.
    """
    m, h, w = images.shape
    kh, kw = kernel.shape

    pad_h = kh // 2
    pad_w = kw // 2

    images_padded = np.pad(images, ((0, 0), (pad_h, pad_h), (pad_w, pad_w)), mode='constant')

    convolved_images = np.zeros_like(images)

    for i in range(m):
        for j in range(h):
            for k in range(w):
                region = images_padded[i, j:j+kh, k:k+kw]
                convolved_value = np.sum(region * kernel)
                convolved_images[i, j, k] = convolved_value
    return convolved_images
