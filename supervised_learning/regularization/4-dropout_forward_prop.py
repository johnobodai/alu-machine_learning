#!/usr/bin/env python3
"""
Implements forward propagation using Dropout.
"""

import numpy as np

def dropout_forward_prop(X, weights, L, keep_prob):
    """
    Forward propagation with Dropout.
    """
    cache = {'A0': X}
    dropout_masks = {}

    for i in range(1, L + 1):
        A_prev = cache['A' + str(i - 1)]
        W = weights['W' + str(i)]
        b = weights['b' + str(i)]
        Z = np.dot(W, A_prev) + b

        if i == L:
            # Last layer, apply softmax activation
            A = np.exp(Z) / np.sum(np.exp(Z), axis=0)
        else:
            # Hidden layers, apply tanh activation and dropout
            A = np.tanh(Z)
            D = np.random.rand(A.shape[0], A.shape[1]) < keep_prob
            A = np.multiply(A, D)
            A /= keep_prob
            dropout_masks['D' + str(i)] = D

        cache['A' + str(i)] = A

    return cache, dropout_masks
