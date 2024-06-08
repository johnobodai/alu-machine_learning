#!/usr/bin/env python3
"""A module that calculates the accuracy of a prediction."""


import tensorflow as tf


def calculate_accuracy(y, y_pred):
    """
    Calculates the accuracy of a prediction.

    Args:
        y: placeholder for the labels of the input data
        y_pred: tensor containing the network’s predictions

    Returns:
        Tensor containing the decimal accuracy of the prediction
    """
    # Check if the predicted labels match the true labels
correct_predictions = tf.equal(tf.argmax(y, axis=1), \
                                tf.argmax(y_pred, axis=1))

    # Calculate the mean accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

    return accuracy
