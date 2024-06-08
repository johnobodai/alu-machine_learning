#!/usr/bin/env python3
"""Creates a tensorflow layer with L2 regularization"""

import tensorflow as tf


def l2_reg_create_layer(prev, n, activation, lambtha):
    """Creates a layer with L2 regularization"""
    kernel_regularizer = tf.keras.regularizers.l2(lambtha)

    layer = tf.keras.layers.Dense(
        units=n,
        activation=activation,
        kernel_regularizer=kernel_regularizer
    )

    return layer(prev)
