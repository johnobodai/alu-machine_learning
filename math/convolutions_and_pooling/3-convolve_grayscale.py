#!/usr/bin/env python3
import numpy as np
"""
Convolution of grayscole images
"""

def convolve_grayscale(images, kernel, padding='same', stride=(1, 1)):
    """
    Performs a convolution on grayscale images.

    Args:
    - images (numpy.ndarray): Input grayscale images with shape (m, h, w).
    - kernel (numpy.ndarray): Convolution kernel with shape (kh, kw).
    - padding (tuple or str): Padding for the height and width of the image.
        - If 'same', performs a same convolution.
        - If 'valid', performs a valid convolution.
        - If a tuple: (ph, pw) where ph is the padding for the height and pw is the padding for the width.
    - stride (tuple): Stride for the height and width of the image.

    Returns:
    - numpy.ndarray: Convolved images.

    """
    m, h, w = images.shape
    kh, kw = kernel.shape
    sh, sw = stride

    # Determine padding
    if padding == 'same':
        ph = int(np.ceil((h - 1) / sh))
        pw = int(np.ceil((w - 1) / sw))
        padding = (ph, pw)
    elif padding == 'valid':
        padding = (0, 0)

    ph, pw = padding

    # Pad the images
    padded_images = np.pad(images, ((0, 0), (ph, ph), (pw, pw)), mode='constant')

    # Calculate output dimensions
    oh = int(((h + 2 * ph - kh) / sh) + 1)
    ow = int(((w + 2 * pw - kw) / sw) + 1)

    # Initialize the result array
    convolved_images = np.zeros((m, oh, ow))

    # Perform convolution
    for i in range(m):
        for j in range(oh):
            for k in range(ow):
                region = padded_images[i, j * sh:j * sh + kh, k * sw:k * sw + kw]
                convolved_images[i, j, k] = np.sum(region * kernel)

    return convolved_images
