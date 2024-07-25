#!/usr/bin/env python3

import tensorflow as tf

def rotate_image(image):
    """
    Rotates an image by 90 degrees counter-clockwise.

    Args:
        image (tf.Tensor): 3D tensor containing the image to rotate.

    Returns:
        tf.Tensor: The rotated image.
    """
    return tf.image.rot90(image)
