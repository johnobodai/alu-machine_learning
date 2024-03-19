#!/usr/bin/env python3
"""Multinormal distribution"""
import numpy as np


class MultiNormal:
    """
    Represents a Multivariate Normal distribution.

    This class calculates and stores the mean and covariance
    matrix of a given data set,
    and provides a method to calculate the PDF at a specific data point.

    Attributes:
        mean (np.ndarray): A numpy array of shape (d, 1).
        cov (np.ndarray): A numpy array of shape (d, d).
    """

    def __init__(self, data: np.ndarray) -> None:
        """
        Initializes a MultiNormal object.

        Args:
            data (np.ndarray): A 2D numpy array of shape (d, n).
                n is the number of data points, d is the number of dimensions.

        Raises:
            TypeError: If data is not a 2D numpy.ndarray.
            ValueError: If data contains less than 2 data points.
        """

        if not isinstance(data, np.ndarray) or data.ndim != 2:
            raise TypeError("data must be a 2D numpy.ndarray")
        if data.shape[0] < 2:
            raise ValueError("data must contain multiple data points")

        n, d = data.shape

        # Calculate the mean
        self.mean = np.mean(data, axis=0, keepdims=True)

        # Center the data
        X_centered = data - self.mean

        # Calculate the covariance matrix (without using numpy.cov)
        self.cov = np.dot(X_centered.T, X_centered) / (n - 1)

    def pdf(self, x: np.ndarray) -> float:
        """
        Calculates the probability density function (PDF) at a data point.

        Args:
            x (np.ndarray): A numpy array of shape (d, 1).
                d is the number of dimensions.

        Returns:
            float: The value of the PDF at the data point x.

        Raises:
            TypeError: If x is not a numpy.ndarray.
            ValueError: If x does not have the shape (d, 1).
        """

        if not isinstance(x, np.ndarray) or x.ndim != 2 or x.shape[1] != 1:
            raise ValueError("x must have the \
                    shape ({}, 1)".format(self.mean.shape[0]))

        d = self.mean.shape[0]  # Get the number of dimensions from mean shape
        n_x = x.shape[0]  # Get the number of data points in x (should be 1)

        if n_x != d:
            raise ValueError("x must have the shape ({}, 1)".format(d))

        # Calculate the mahalanobis distance
        mahalanobis = np.linalg.inv(self.cov) @ (x - self.mean)
        mahalanobis_sq = np.dot(mahalanobis.T, mahalanobis)

        # Constant term (excluding normalization constant)
        constant_term = -0.5 * mahalanobis_sq

        # Calculate the normalization constant
        det_cov = np.linalg.det(self.cov)

        # PDF formula (without numpy.exp for efficiency)
        pdf = np.exp(constant_term) / (np.sqrt((2 * np.pi) ** d * det_cov))

        return pdf[0, 0]  # Return the only element for a single data point
