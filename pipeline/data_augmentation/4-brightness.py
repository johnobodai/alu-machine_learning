#!/usr/bin/env python3

import tensorflow as tf

def change_brightness(image, max_delta):
    """
    Randomly changes the brightness of an image.

    Args:
        image (tf.Tensor): 3D tensor containing the image to change.
        max_delta (float): The maximum amount the image should be brightened (or darkened).

    Returns:
        tf.Tensor: The altered image.
    """
    return tf.image.random_brightness(image, max_delta)
