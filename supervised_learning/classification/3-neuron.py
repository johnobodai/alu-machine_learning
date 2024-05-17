#!/usr/bin/env python3
"""
Module for a single neuron performing binary
classification.
"""

import numpy as np

class Neuron:
    """
    Represents a single neuron performing binary
    classification.
    """

    def __init__(self, nx):
        """
        Initializes a Neuron instance.

        Parameters:
            nx (int): Number of input features.

        Raises:
            TypeError: If nx is not an integer.
            ValueError: If nx is less than 1.
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be positive")

        self.__W = np.random.randn(1, nx)
        self.__b = 0
        self.__A = 0

    def forward_prop(self, X):
        """
        Calculates forward propagation.

        Parameters:
            X (numpy.ndarray): Input data (shape: (nx, m)).

        Returns:
            numpy.ndarray: Activated output of the
            neuron.
        """
        Z = np.dot(self.__W, X) + self.__b
        self.__A = 1 / (1 + np.exp(-Z))
        return self.__A

    def cost(self, Y, A):
        """
        Calculates the cost using logistic regression.

        Parameters:
            Y (numpy.ndarray): Correct labels (shape:
            (1, m)).
            A (numpy.ndarray): Activated output (shape:
            (1, m)).

        Returns:
            float: Cost of the model.
        """
        m = Y.shape[1]
        cost = -np.sum((Y * np.log(A)) +
                       ((1 - Y) * np.log(1.0000001 - A))) / m
        return cost

    @property
    def W(self):
        """
        Getter for the weights vector.

        Returns:
            numpy.ndarray: Weights vector.
        """
        return self.__W

    @property
    def b(self):
        """
        Getter for the bias.

        Returns:
            float: Bias.
        """
        return self.__b

    @property
    def A(self):
        """
        Getter for the activated output.

        Returns:
            float: Activated output.
        """
        return self.__A
