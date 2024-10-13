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
            vocab (int): Size of the output vocabulary.
            embedding (int): Dimensionality of the embedding vector.
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
        Decode the input.

        Args:
            x (Tensor): Previous word in the target sequence.
            s_prev (Tensor): Previous decoder hidden state.
            hidden_states (Tensor): Encoder outputs.

        Returns:
            Tuple of (y, s).
        """
        # Get the embedding for the input word
        x = self.embedding(x)  # Shape: (batch, 1, embedding)

        # Use self-attention to get the context vector
        attention = SelfAttention(s_prev.shape[1])
        context, _ = attention(s_prev, hidden_states)  # context shape: (batch, units)

        # Reshape context to (batch, 1, units) for concatenation
        context = tf.expand_dims(context, axis=1)  # Shape: (batch, 1, units)

        # Concatenate context with x
        x = tf.concat([context, x], axis=-1)  # Shape: (batch, 1, units + embedding)

        # Pass through GRU
        output, s = self.gru(x)  # output shape: (batch, 1, units)

        # Get the final output word
        y = self.F(output)  # y shape: (batch, 1, vocab)
        y = tf.squeeze(y, axis=1)  # Shape: (batch, vocab)

        return y, s
