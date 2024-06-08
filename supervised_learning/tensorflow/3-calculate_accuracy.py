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
    a_pred = tf.arg_max(y_pred, 1)
    b_inp = tf.arg_max(y, 1)
    output = tf.equal(b_inp, a_pred)
    return tf.reduce_mean(tf.cast(output, tf.float32))
