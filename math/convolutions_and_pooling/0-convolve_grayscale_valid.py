#!/usr/bin/env python3
import numpy as np

"""
Module: convolve_grayscale_valid

This module provides a function to perform a valid convolution on grayscale images.
"""

def convolve_grayscale_valid(images, kernel):
    """
    Perform a valid convolution on grayscale images.

    Args:
        images (array): multiple grayscale images
        kernel (array): the kernel for the convolution

    Returns:
        array: the convolved image
    """
    # Dimension of images and kernel.
    m, h, w = images.shape
    kh, kw = kernel.shape

    # Shape of the output image.
    oh = h - kh + 1  # Output height
    ow = w - kw + 1  # Output width

    convolved_images = np.zeros((m, oh, ow))

    # For each image.
    for i in range(m):
        # For each pixel in image.
        for j in range(oh):
            for k in range(ow):
                region = images[i, j:j+kh, k:k+kw]
                convolved_value = np.sum(region * kernel)
                convolved_images[i, j, k] = convolved_value

    return convolved_images
