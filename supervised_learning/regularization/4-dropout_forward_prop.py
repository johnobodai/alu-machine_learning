#!/usr/bin/env python3
"""
Implements forward propagation using Dropout.
"""

import numpy as np


def dropout_forward_prop(X, weights, L, keep_prob):
    """
    Forward propagation with Dropout.
    """
    # Initialize dictionary to store layer outputs and dropout masks
    cache = {}
    # Assign input data to the first layer output
    cache['A0'] = X

    # Loop through each layer
    for i in range(1, L + 1):
        # Retrieve previous layer output
        A_prev = cache['A' + str(i - 1)]
        # Retrieve weights and biases
        W = weights['W' + str(i)]
        b = weights['b' + str(i)]
        # Calculate Z value for the current layer
        Z = np.dot(W, A_prev) + b
        # Apply activation function (tanh for hidden layers, softmax for output layer)
        if i == L:
            A = np.exp(Z) / np.sum(np.exp(Z), axis=0, keepdims=True)
        else:
            A = np.tanh(Z)
            # Apply dropout
            D = np.random.rand(A.shape[0], A.shape[1]) < keep_prob
            A = np.multiply(A, D) / keep_prob
            cache['D' + str(i)] = D
        # Store layer output in cache
        cache['A' + str(i)] = A

    return cache
