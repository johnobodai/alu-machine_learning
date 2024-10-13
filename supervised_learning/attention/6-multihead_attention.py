#!/usr/bin/env python3
"""
Defines a class for multi-head attention
"""

import tensorflow as tf
sdp_attention = __import__('5-sdp_attention').sdp_attention


class MultiHeadAttention(tf.keras.layers.Layer):
    """
    Class for performing multi-head attention
    """

    def __init__(self, dm, h):
        """
        Initializes the MultiHeadAttention class

        Args:
            dm (int): Dimensionality of the model
            h (int): Number of heads
        """
        super(MultiHeadAttention, self).__init__()
        self.h = h
        self.dm = dm
        self.depth = dm // h
        self.Wq = tf.keras.layers.Dense(units=dm)
        self.Wk = tf.keras.layers.Dense(units=dm)
        self.Wv = tf.keras.layers.Dense(units=dm)
        self.linear = tf.keras.layers.Dense(units=dm)

    def split_heads(self, tensor, batch_size):
        """
        Splits the last dimension into multiple heads

        Args:
            tensor: Input tensor to split
            batch_size: Batch size of the input

        Returns:
            Tensor: Reshaped and transposed tensor
        """
        tensor = tf.reshape(tensor, (batch_size, -1, self.h, self.depth))
        return tf.transpose(tensor, perm=[0, 2, 1, 3])

    def call(self, Q, K, V, mask):
        """
        Computes scaled dot product attention

        Args:
            Q: Query tensor
            K: Key tensor
            V: Value tensor
            mask: Optional mask (always None)

        Returns:
            Tensor: Attention output
            Tensor: Attention weights
        """
        batch_size = tf.shape(Q)[0]
        query = self.Wq(Q)
        key = self.Wk(K)
        value = self.Wv(V)

        query = self.split_heads(query, batch_size)
        key = self.split_heads(key, batch_size)
        value = self.split_heads(value, batch_size)

        attention, weights = sdp_attention(query, key, value, mask)

        attention = tf.transpose(attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(attention, (batch_size, -1, self.dm))
        outputs = self.linear(concat_attention)

        return outputs, weights

