#!/usr/bin/env python3
"""Class MultiNormal representing a Multivariate Normal distribution"""
import numpy as np


class MultiNormal:
    """Class that represents a Multivariate Normal distribution"""

    def __init__(self, data):
        """Constructor for MultiNormal class

        Args:
            data (numpy.ndarray): A 2D numpy array of shape (d, n).

        Raises:
            TypeError: If data is not a 2D numpy.ndarray.
            ValueError: If data contains fewer than 2 data points.
        """
        if not isinstance(data, np.ndarray) or len(data.shape) != 2:
            raise TypeError('data must be a 2D numpy.ndarray')

        d, n = data.shape
        if n < 2:
            raise ValueError("data must contain multiple data points")

        self.mean, self.cov = self.mean_cov(data)

    @staticmethod
    def mean_cov(X):
        """Calculates the mean and covariance of a data set

        Args:
            X (numpy.ndarray): A 2D numpy array of shape (d, n)

        Returns:
            tuple: A tuple containing the mean and covariance

        """
        d, n = X.shape
        m = np.mean(X, axis=1, keepdims=True)
        C = np.dot((X - m), (X - m).T) / (n - 1)
        return m, C

    def pdf(self, x):
        """Calculates the PDF at a data point

        Args:
            x (numpy.ndarray): A numpy array of shape (d, 1)

        Returns:
            float: The value of the PDF at the data point x.

        Raises:
            TypeError: If x is not a numpy.ndarray.
            ValueError: If x does not have the shape (d, 1).

        """
        if not isinstance(x, np.ndarray):
            raise TypeError("x must be a numpy.ndarray")

        d = self.cov.shape[0]
        if x.shape != (d, 1):
            raise ValueError('x must have the shape ({}, 1)'.format(d))

        m = self.mean
        cov = self.cov
        bottom = np.sqrt(((2 * np.pi) ** d) * (np.linalg.det(cov)))
        inv = np.linalg.inv(cov)
        exp = (-0.5 * np.matmul(np.matmul((x - m).T, inv), (x - m)))
        result = (1 / bottom) * np.exp(exp[0][0])
        return result
