#!/usr/bin/env python3
"""A module that calculates the cost of a neural network with L2 regularization"""

import numpy as np

def l2_reg_cost(cost, lambtha, weights, L, m):
    """
    Calculates the cost of a neural network with L2 regularization.

    Args:
        cost: the cost of the network without L2 regularization
        lambtha: the regularization parameter
        weights: a dictionary of the weights and biases (numpy.ndarrays) of the neural network
        L: the number of layers in the neural network
        m: the number of data points used

    Returns:
        The cost of the network accounting for L2 regularization
    """
    l2_cost = cost
    l2_sum = 0

    for i in range(1, L + 1):
        l2_sum += np.sum(np.square(weights[f'W{i}']))

    l2_cost += (lambtha / (2 * m)) * l2_sum
    return l2_cost
