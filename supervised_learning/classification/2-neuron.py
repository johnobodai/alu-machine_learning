#!/usr/bin/env python3
import numpy as np


class Neuron:
    """
    Class that defines a single neuron performing binary classification.
    """

    def __init__(self, nx):
        """
        Initializes a Neuron instance.

        Parameters:
            nx (int): The number of input features to the neuron.

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
        Calculates the forward propagation of the neuron.

        Parameters:
            X (numpy.ndarray): Input data (shape: (nx, m)).

        Returns:
            numpy.ndarray: The activated output of the neuron.
        """
        Z = np.dot(self.__W, X) + self.__b
        self.__A = 1 / (1 + np.exp(-Z))
        return self.__A

    @property
    def W(self):
        """
        Getter method for the weights vector of the neuron.

        Returns:
            numpy.ndarray: The weights vector.
        """
        return self.__W

    @property
    def b(self):
        """
        Getter method for the bias of the neuron.

        Returns:
            float: The bias.
        """
        return self.__b

    @property
    def A(self):
        """
        Getter method for the activated output of the neuron.

        Returns:
            float: The activated output.
        """
        return self.__A