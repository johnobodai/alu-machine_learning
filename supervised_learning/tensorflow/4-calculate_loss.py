#!/usr/bin/env python3
"""A module that calculates the softmax cross-entropy loss of a prediction."""


import tensorflow as tf


def calculate_loss(y, y_pred):
    """
    Calculates the softmax cross-entropy loss of a prediction.

    Args:
        y: placeholder for the labels of the input data
        y_pred: tensor containing the network’s predictions

    Returns:
        Tensor containing the loss of the prediction
    """
    # Calculate the softmax cross-entropy loss
    loss = tf.losses.softmax_cross_entropy(onehot_labels=y, logits=y_pred)

    return loss
