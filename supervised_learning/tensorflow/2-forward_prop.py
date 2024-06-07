#!/usr/bin/env python3
"""A module that creates the forward propagation graph for a neural network."""


import tensorflow as tf
create_layer = __import__('1-create_layer').create_layer


def forward_prop(x, layer_sizes=[], activations=[]):
    """
    Creates the forward propagation graph for the neural network.

    Args:
        x: placeholder for the input data
        layer_sizes: list containing the number of nodes in each layer
        activations: list containing the activation functions for each layer

    Returns:
        Prediction of the network in tensor form
    """
    prev = x
    for i in range(len(layer_sizes)):
        prev = create_layer(prev, layer_sizes[i], activations[i])
    return prev
