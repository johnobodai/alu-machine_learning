#!/usr/bin/env python3

import tensorflow as tf

def crop_image(image, size):
    """
    Performs a random crop of an image.

    Args:
        image (tf.Tensor): 3D tensor containing the image to crop.
        size (tuple): Size of the crop (height, width, channels).

    Returns:
        tf.Tensor: The cropped image.
    """
    return tf.image.random_crop(image, size)
