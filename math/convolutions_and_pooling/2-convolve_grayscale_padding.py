#!/usr/bin/env python3
import numpy as np
"""
Performs a convolution on grayscale images with custom padding
"""

def convolve_grayscale_padding(images, kernel, padding):
    """
    Performs a convolution on grayscale images with custom padding.

    Args:
    - images (numpy.ndarray): Input grayscale images with shape (m, h, w).
    - kernel (numpy.ndarray): Convolution kernel with shape (kh, kw).
    - padding (tuple): Padding for the height and width of the image.

    Returns:
    - numpy.ndarray: Convolved images.

    """
    m, h, w = images.shape
    kh, kw = kernel.shape
    ph, pw = padding

    # Pad the images
    padded_images = np.pad(images, ((0, 0), (ph, ph), (pw, pw)), mode='constant')

    # Initialize the result array
    convolved_images = np.zeros((m, h, w))

    # Perform convolution
    for i in range(m):
        for j in range(h):
            for k in range(w):
                region = padded_images[i, j:j+kh, k:k+kw]
                convolved_images[i, j, k] = np.sum(region * kernel)

    return convolved_images

