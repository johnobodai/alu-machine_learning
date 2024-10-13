#!/usr/bin/env python3
"""
RNN Decoder using GRU for machine translation
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
            vocab (int): Vocabulary size.
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
            x (Tensor): Input tensor.
            s_prev (Tensor): Previous decoder hidden state.
            hidden_states (Tensor): Outputs of the encoder.

        Returns:
            Tuple of (y, s).
            y (Tensor): Output word as one hot vector in the target vocabulary.
            s (Tensor): New decoder hidden state.
        """
        attention = SelfAttention(s_prev.shape[1])
        context, weights = attention(s_prev, hidden_states)
        x = self.embedding(x)
        context = tf.expand_dims(context, 1)
        x = tf.concat([context, x], axis=-1)
        output, state = self.gru(x)
        output = tf.reshape(output, (-1, output.shape[2]))
        y = self.F(output)
        return y, state
