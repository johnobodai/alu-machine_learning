#!/usr/bin/env python3
"""A module that creates a layer for a neural network."""


import tensorflow as tf


def create_layer(prev, n, activation):
    """
    Creates a layer for a neural network.

    Args:
        prev: tensor output of the previous layer
        n: number of nodes in the layer to create
        activation: activation function for the layer

    Returns:
        Tensor output of the layer
    """
    init = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    layer = tf.layers.Dense(units=n, activation=activation,
                            kernel_initializer=init, name='layer')
    return layer(prev)
