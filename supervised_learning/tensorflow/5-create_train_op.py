#!/usr/bin/env python3
"""A module that creates the training operation for the network."""


import tensorflow as tf


def create_train_op(loss, alpha):
    """
    Creates the training operation for the network.

    Args:
        loss: the loss of the network’s prediction
        alpha: the learning rate

    Returns:
        An operation that trains the network using gradient descent
    """
    # Create an optimizer using gradient descent
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=alpha)

    # Minimize the loss using the optimizer
    train_op = optimizer.minimize(loss)

    return train_op
