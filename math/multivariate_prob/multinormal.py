#!/usr/bin/env python3
"""
Class MultiNormal representing a Multivariate Normal distribution.
"""

import numpy as np


class MultiNormal:
    """
    Class that represents a Multivariate Normal distribution.
    """

    def __init__(self, data):
        """
        Constructor for MultiNormal class.

        Args:
            data (numpy.ndarray): A 2D numpy array of shape (n, d)

        Raises:
            TypeError: If data is not a 2D numpy.ndarray.
            ValueError: If n is less than 2.
        """
        if type(data) is not np.ndarray or len(data.shape) != 2:
            raise TypeError("data must be a 2D numpy.ndarray")
        d, n = data.shape
        if n < 2:
            raise ValueError("data must contain multiple data points")

        self.data = data
        mean = np.mean(data, axis=1, keepdims=True)
        self.mean = mean
        cov = np.matmul(data - mean, data.T - mean.T) / (n - 1)
        self.cov = cov

    def pdf(self, x):
        """
        Calculates the PDF at a data point.

        Args:
            x (numpy.ndarray): A numpy array of shape (d, 1)

        Returns:
            float: The PDF at the given data point.
        """
        if type(x) is not np.ndarray:
            raise TypeError("x must be a numpy.ndarray")
        d = self.cov.shape[0]
        if len(x.shape) != 2:
            raise ValueError("x must have the shape ({}, 1)".format(d))
        test_d, one = x.shape
        if test_d != d or one != 1:
            raise ValueError("x must have the shape ({}, 1)".format(d))

        det = np.linalg.det(self.cov)
        inv = np.linalg.inv(self.cov)
        pdf = 1.0 / np.sqrt(((2 * np.pi) ** d) * det)
        mult = np.matmul(np.matmul((x - self.mean).T, inv), (x - self.mean))
        pdf *= np.exp(-0.5 * mult)
        pdf = pdf[0][0]
        return pdf
