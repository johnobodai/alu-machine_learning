#!/usr/bin/env python3
"""
Updates the weights of a neural network
using L2 regularization with gradient descent
"""

import numpy as np

def l2_reg_gradient_descent(Y, weights, cache,
                            alpha, lambtha, L):
    """
    Updates weights and biases of a neural network
    using gradient descent with L2 regularization.
    """
    m = Y.shape[1]
    A_prev = cache['A{}'.format(L)]
    dZ = A_prev - Y

    for i in range(L, 0, -1):
        A_prev = cache['A{}'.format(i-1)]
        W = weights['W{}'.format(i)]
        b = weights['b{}'.format(i)]

        dW = (np.matmul(dZ, A_prev.T) / m) + \
             (lambtha * W / m)
        db = np.sum(dZ, axis=1, keepdims=True) / m

        if i > 1:
            dA_prev = 1 - np.square(A_prev)
            dZ = np.matmul(W.T, dZ) * dA_prev

        weights['W{}'.format(i)] -= alpha * dW
        weights['b{}'.format(i)] -= alpha * db
