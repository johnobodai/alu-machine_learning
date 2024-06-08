#!/usr/bin/env python3

"""Builds, trains, and saves a neural network model."""

import numpy as np
import tensorflow as tf


def model(Data_train, Data_valid, layers, activations,
          alpha=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8,
          decay_rate=1, batch_size=32, epochs=5,
          save_path='/tmp/model.ckpt'):
    """
    Args:
        Data_train (tuple): Training inputs and labels.
        Data_valid (tuple): Validation inputs and labels.
        layers (list): Number of nodes in each layer.
        activations (list): Activation functions for each layer.
        alpha (float): Learning rate.
        beta1 (float): Weight for first moment of Adam
        Optimization.
        beta2 (float): Weight for second moment of Adam
        Optimization.
        epsilon (float): Small number to avoid division by
        zero.
        decay_rate (float): Decay rate for inverse time decay
        of learning rate.
        batch_size (int): Number of data points in a
        mini-batch.
        epochs (int): Number of times to pass through dataset.
        save_path (str): Path to save the model.

    Returns:
        str: Path where the model was saved.
    """

    tf.reset_default_graph()

    # Unpack training and validation data
    X_train, Y_train = Data_train
    X_valid, Y_valid = Data_valid

    # Placeholders for input data
    input_shape = X_train.shape[1]
    output_shape = Y_train.shape[1]
    X = tf.placeholder(tf.float32, shape=(None, input_shape), name='X')
    Y = tf.placeholder(tf.float32, shape=(None, output_shape), name='Y')
    is_training = tf.placeholder(tf.bool, name='is_training')

    # Create neural network
    def dense_layer(x, units, activation, name):
        with tf.name_scope(name):
            layer = tf.layers.dense(x, units=units, activation=None, name='dense')
            if activation:
                layer = activation(layer, name='activation')
            return layer

    with tf.name_scope('neural_network'):
        previous_layer = X
        for i, (layer_size, activation_fn) in enumerate(zip(layers, activations)):
            previous_layer = dense_layer(previous_layer, layer_size, activation_fn, name=f'layer_{i}')

        logits = previous_layer

    # Loss function
    with tf.name_scope('loss'):
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=logits))

    # Optimizer with learning rate decay
    with tf.name_scope('train'):
        global_step = tf.Variable(0, trainable=False)
        decayed_learning_rate = tf.train.inverse_time_decay(alpha, global_step, decay_rate, 1)
        optimizer = tf.train.AdamOptimizer(learning_rate=decayed_learning_rate, beta1=beta1, beta2=beta2, epsilon=epsilon)
        train_op = optimizer.minimize(loss, global_step=global_step)

    # Evaluation metrics
    with tf.name_scope('accuracy'):
        correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(Y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    # Initialize variables and create saver
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    # Training
    with tf.Session() as sess:
        sess.run(init)

        # Training loop
        for epoch in range(epochs):
            print(f'After {epoch} epochs:')
            # Shuffle training data
            indices = np.random.permutation(len(X_train))
            X_train_shuffled = X_train[indices]
            Y_train_shuffled = Y_train[indices]

            # Mini-batch gradient descent
            for i in range(0, len(X_train_shuffled), batch_size):
                X_batch = X_train_shuffled[i:i+batch_size]
                Y_batch = Y_train_shuffled[i:i+batch_size]

                _, step, batch_loss, batch_accuracy = sess.run([train_op, global_step, loss, accuracy],
                                                                feed_dict={X: X_batch, Y: Y_batch, is_training: True})
                if (i // batch_size) % 100 == 0:
                    print(f'\tStep {i // batch_size}:')
                    print(f'\t\tCost: {batch_loss}')
                    print(f'\t\tAccuracy: {batch_accuracy}')

            # Print training and validation stats after each epoch
            train_cost, train_accuracy = sess.run([loss, accuracy], feed_dict={X: X_train, Y: Y_train, is_training: False})
            valid_cost, valid_accuracy = sess.run([loss, accuracy], feed_dict={X: X_valid, Y: Y_valid, is_training: False})
            print(f'\tTraining Cost: {train_cost}')
            print(f'\tTraining Accuracy: {train_accuracy}')
            print(f'\tValidation Cost: {valid_cost}')
            print(f'\tValidation Accuracy: {valid_accuracy}')

        # Save the model
        save_path = saver.save(sess, save_path)
        print(f'Model saved in path: {save_path}')

    return save_path
