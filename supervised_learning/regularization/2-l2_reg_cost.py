#!/usr/bin/env python3
"""
Updates the weights of a neural network
using L2 regularization with gradient descent
"""

import tensorflow as tf

def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
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

        dW = (tf.matmul(dZ, A_prev, transpose_b=True) / m) + \
             (lambtha * W / m)
        db = tf.reduce_sum(dZ, axis=1, keepdims=True) / m

        if i > 1:
            dA_prev = 1 - tf.square(A_prev)
            dZ = tf.matmul(W, dZ, transpose_a=True) * dA_prev

        weights['W{}'.format(i)].assign_sub(alpha * dW)
        weights['b{}'.format(i)].assign_sub(alpha * db)

