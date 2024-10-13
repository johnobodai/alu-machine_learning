#!/usr/bin/env python3
"""
Self Attention for machine translation.
"""

import tensorflow as tf


class SelfAttention(tf.keras.layers.Layer):
    """
    Self Attention layer for machine translation.
    """

    def __init__(self, units):
        """
        Initialize the self-attention layer.

        Args:
            units (int): Number of hidden units.
        """
        super(SelfAttention, self).__init__()
        self.W = tf.keras.layers.Dense(units)
        self.U = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, s_prev, hidden_states):
        """
        Calculate the context and attention weights.

        Args:
            s_prev (Tensor): Previous decoder hidden state.
            hidden_states (Tensor): Encoder outputs.

        Returns:
            Tuple of (context, weights).
        """
        # Calculate attention scores
        score = self.V(tf.nn.tanh(self.W(s_prev) + self.U(hidden_states)))  # (batch, input_seq_len, 1)

        # Get weights by applying softmax to the scores
        weights = tf.nn.softmax(score, axis=1)  # (batch, input_seq_len, 1)

        # Calculate context vector as weighted sum of hidden states
        context = tf.reduce_sum(weights * hidden_states, axis=1)  # (batch, units)

        return context, weights
