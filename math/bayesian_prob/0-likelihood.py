#!/usr/bin/env python3
"""
Module for calculating likelihood in Bayesian statistics.
"""
import numpy as np


def binomial_coefficient(n, k):
    """
    Calculate the binomial coefficient (n choose k) using dynamic programming.
    """
    if k > n:
        return 0
    dp = [0] * (k + 1)
    dp[0] = 1
    for i in range(1, n + 1):
        for j in range(min(i, k), 0, -1):
            dp[j] += dp[j - 1]
    return dp[k]


def likelihood(x, n, P):
    """
    Calculate the likelihood of obtaining the data given
    various hypothetical probabilities.

    Parameters:
        x (int): The number of patients that develop
                 severe side effects.
        n (int): The total number of patients observed.
        P (numpy.ndarray): An array containing the various
                           hypothetical probabilities of developing
                           severe side effects.

    Returns:
        numpy.ndarray: An array containing the likelihood
        of obtaining the data, x and n, for each probability in P.

    Raises:
        ValueError: If n is not a positive integer, x is not an
        integer that is greater than or equal to 0, x is greater than n,
                    P is not a 1D numpy array, or any value in P is
                    not in the range [0, 1].
        TypeError: If P is not a 1D numpy array.
    """
    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer")

    if not isinstance(x, int) or x < 0:
        raise ValueError("x must be an integer that is "
                         "greater than or equal to 0")

    if x > n:
        raise ValueError("x cannot be greater than n")

    if not isinstance(P, np.ndarray) or P.ndim != 1:
        raise TypeError("P must be a 1D numpy.ndarray")

    if np.any((P < 0) | (P > 1)):
        raise ValueError("All values in P must be in "
                         "the range [0, 1]")

    likelihoods = np.array([binomial_coefficient(n, x) * (p ** x) *
                            ((1 - p) ** (n - x)) for p in P])

    return likelihoods
