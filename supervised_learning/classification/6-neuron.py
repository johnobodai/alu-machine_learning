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

    def evaluate(self, X, Y):
        """
        Evaluates the neuron's predictions.

        Parameters:
            X (numpy.ndarray): Input data (shape: (nx, m)).
            Y (numpy.ndarray): Correct labels (shape: (1, m)).

        Returns:
            tuple: Prediction and cost of the network.
                - Prediction: numpy.ndarray with shape (1, m)
                - Cost: Cost of the network.
        """
        A = self.forward_prop(X)
        prediction = np.where(A >= 0.5, 1, 0)
        cost = self.cost(Y, A)
        return prediction, cost

    def gradient_descent(self, X, Y, A, alpha=0.05):
        """
        Calculates one pass of gradient descent on the neuron.

        Parameters:
            X (numpy.ndarray): Input data (shape: (nx, m)).
            Y (numpy.ndarray): Correct labels (shape: (1, m)).
            A (numpy.ndarray): Activated output (shape: (1, m)).
            alpha (float): Learning rate.

        """
        m = Y.shape[1]

        dz = A - Y
        dw = np.dot(X, dz.T) / m
        db = np.sum(dz) / m

        self.__W -= alpha * dw.T
        self.__b -= alpha * db

    def train(self, X, Y, iterations=5000, alpha=0.05, verbose=True, graph=True, step=100):
        """
        Trains the neuron.

        Parameters:
            X (numpy.ndarray): Input data (shape: (nx, m)).
            Y (numpy.ndarray): Correct labels (shape: (1, m)).
            iterations (int): Number of iterations to train over.
            alpha (float): Learning rate.
            verbose (bool): Whether to print information about the training.
            graph (bool): Whether to graph information about the training.
            step (int): Step for printing and plotting.

        Returns:
            tuple: Evaluation of the training data after iterations of training have occurred.
        """
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations < 1:
            raise ValueError("iterations must be a positive integer")
        if not isinstance(alpha, float):
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        if not isinstance(step, int):
            raise TypeError("step must be an integer")
        if step <= 0 or step > iterations:
            raise ValueError("step must be positive and <= iterations")

        costs = []
        for i in range(iterations + 1):
            A = self.forward_prop(X)
            self.gradient_descent(X, Y, A, alpha)

            if verbose and i % step == 0:
                cost = self.cost(Y, A)
                print("Cost after {} iterations: {}".format(i, cost))
                costs.append(cost)

        if graph:
            plt.plot(costs, 'b-')
            plt.xlabel('Iteration')
            plt.ylabel('Cost')
            plt.title('Training Cost')
            plt.show()

        return self.evaluate(X, Y)

