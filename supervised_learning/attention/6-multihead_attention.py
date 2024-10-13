#!/usr/bin/env python3
"""
Implements a class inheriting from
tensorflow.keras.layers.Layer to execute
multi-head attention.
"""

import tensorflow as tf
scaled_attention = __import__('5-sdp_attention').sdp_attention


class MultiHeadAttention(tf.keras.layers.Layer):
    """
    Class to implement multi-head attention.
    """
    def __init__(self, model_dim, num_heads):
        """
        Class constructor.

        Parameters:
            model_dim [int]: Dimensionality of the model.
            num_heads [int]: Number of attention heads.
        """
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.model_dim = model_dim
        self.head_dim = model_dim // num_heads
        self.dense_query = tf.keras.layers.Dense(units=model_dim)
        self.dense_key = tf.keras.layers.Dense(units=model_dim)
        self.dense_value = tf.keras.layers.Dense(units=model_dim)
        self.output_dense = tf.keras.layers.Dense(units=model_dim)

    def split_heads(self, x, batch_size):
        """
        Splits last dimension into (num_heads, head_dim).
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, 
                           self.head_dim))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, queries, keys, values, mask):
        """
        Generates attention output and weights.

        Parameters:
            queries: Input for query matrix.
            keys: Input for key matrix.
            values: Input for value matrix.
            mask: Optional mask (default None).

        Returns:
            outputs: Scaled dot product attention.
            weights: Attention weights.
        """
        batch_size = tf.shape(queries)[0]

        query_matrix = self.dense_query(queries)
        key_matrix = self.dense_key(keys)
        value_matrix = self.dense_value(values)

        query_matrix = self.split_heads(query_matrix, 
                                         batch_size)
        key_matrix = self.split_heads(key_matrix, 
                                       batch_size)
        value_matrix = self.split_heads(value_matrix, 
                                         batch_size)

        attention_output, attention_weights = scaled_attention(
            query_matrix, key_matrix, value_matrix, mask)

        attention_output = tf.transpose(attention_output, 
                                         perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(attention_output, 
                                      (batch_size, -1, self.model_dim))
        outputs = self.output_dense(concat_attention)

        return outputs, attention_weights
