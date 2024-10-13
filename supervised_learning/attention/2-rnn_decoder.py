#!/usr/bin/env python3
"""
RNN Decoder for machine translation.
"""

import tensorflow as tf
SelfAttention = __import__('1-self_attention').SelfAttention

class RNNDecoder(tf.keras.layers.Layer):
    """
    RNN Decoder using GRU for machine translation.
    """

    def __init__(self, vocab, embedding, units, batch):
        """
        Initialize the decoder.

        Args:
            vocab (int): Output vocabulary size.
            embedding (int): Embedding vector size.
            units (int): Number of GRU units.
            batch (int): Batch size.
        """
        super(RNNDecoder, self).__init__()
        self.embedding = tf.keras.layers.Embedding(vocab, embedding)
        self.gru = tf.keras.layers.GRU(
            units,
            return_sequences=True,
            return_state=True,
            recurrent_initializer='glorot_uniform'
        )
        self.F = tf.keras.layers.Dense(vocab)

    def call(self, x, s_prev, hidden_states):
        """
        Run the decoder.

        Args:
            x (Tensor): Input tensor of shape (batch, 1).
            s_prev (Tensor): Previous decoder hidden state of shape (batch, units).
            hidden_states (Tensor): Encoder hidden states of shape (batch, input_seq_len, units).

        Returns:
            Tuple of (y, s).
        """
        x = self.embedding(x)
        attention = SelfAttention(256)  # Assuming units=256 for SelfAttention
        context, _ = attention(s_prev, hidden_states)
        x = tf.concat([context, x], axis=-1)  # Concatenate context with the embedding
        output, s = self.gru(x, initial_state=s_prev)  # GRU layer
        y = self.F(output)  # Final dense layer
        return y, s

