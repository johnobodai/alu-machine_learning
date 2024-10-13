#!/usr/bin/env python3
"""
Self Attention for machine translation.
"""


import tensorflow as tf


class SelfAttention(tf.keras.layers.Layer):
    """
    Self Attention mechanism for machine translation.
    """

    def __init__(self, units):
        """
        Initialize the attention layers.
        Args:
            units (int): Number of hidden units.
        """
        super(SelfAttention, self).__init__()
        self.W = tf.keras.layers.Dense(units)
        self.U = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, s_prev, hidden_states):
        """
        Compute the context and attention weights.
        Args:
            s_prev (Tensor): Previous decoder hidden state of shape (batch, units).
            hidden_states (Tensor): Encoder hidden states of shape (batch, input_seq_len, units).
        Returns:
            Tuple of (context, weights).
        """
        w = self.W(tf.expand_dims(s_prev, 1))
        u = self.U(hidden_states)
        v = self.V(tf.nn.tanh(w + u))
        weights = tf.nn.softmax(v, axis=1)
        context = tf.reduce_sum(weights * hidden_states, axis=1)
        return context, weights
