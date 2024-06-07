#!/usr/bin/env python3
"""A module that builds, trains, and saves a neural network classifier."""


import numpy as np
import tensorflow as tf
calculate_accuracy = __import__('3-calculate_accuracy').calculate_accuracy
calculate_loss = __import__('4-calculate_loss').calculate_loss
create_placeholders = __import__('0-create_placeholders').create_placeholders
create_train_op = __import__('5-create_train_op').create_train_op
forward_prop = __import__('2-forward_prop').forward_prop


def train(X_train, Y_train, X_valid, Y_valid, layer_sizes, activations,
          alpha, iterations, save_path="/tmp/model.ckpt"):
    """
    Builds, trains, and saves a neural network classifier.

    Args:
        X_train: numpy.ndarray containing the training input data
        Y_train: numpy.ndarray containing the training labels
        X_valid: numpy.ndarray containing the validation input data
        Y_valid: numpy.ndarray containing the validation labels
        layer_sizes: list containing the number of nodes in each layer
                     of the network
        activations: list containing the activation functions for each layer
                     of the network
        alpha: learning rate
        iterations: number of iterations to train over
        save_path: where to save the model

    Returns:
        Path where the model was saved
    """
    tf.set_random_seed(0)

    x, y = create_placeholders(X_train.shape[1], Y_train.shape[1])
    y_pred = forward_prop(x, layer_sizes, activations)
    loss = calculate_loss(y, y_pred)
    accuracy = calculate_accuracy(y, y_pred)
    train_op = create_train_op(loss, alpha)

    # Add tensors and operations to the graph's collection
    tf.add_to_collection('x', x)
    tf.add_to_collection('y', y)
    tf.add_to_collection('y_pred', y_pred)
    tf.add_to_collection('loss', loss)
    tf.add_to_collection('accuracy', accuracy)
    tf.add_to_collection('train_op', train_op)

    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for i in range(iterations + 1):
            train_feed_dict = {x: X_train, y: Y_train}
            valid_feed_dict = {x: X_valid, y: Y_valid}

            if i % 100 == 0:
                train_loss, train_accuracy = sess.run([loss, accuracy],
                                                      feed_dict=train_feed_dict)
                valid_loss, valid_accuracy = sess.run([loss, accuracy],
                                                      feed_dict=valid_feed_dict)
                print("After {} iterations:".format(i))
                print("\tTraining Cost: {}".format(train_loss))
                print("\tTraining Accuracy: {}".format(train_accuracy))
                print("\tValidation Cost: {}".format(valid_loss))
                print("\tValidation Accuracy: {}".format(valid_accuracy))

            sess.run(train_op, feed_dict=train_feed_dict)

        save_path = saver.save(sess, save_path)
        print("Model saved in path: {}".format(save_path))

    return save_path
