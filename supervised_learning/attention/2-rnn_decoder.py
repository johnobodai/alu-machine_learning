#!/usr/bin/env python3
"""
RNN Decoder using GRU for machine translation.
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
            hidden_states (Tensor): Encoder's hidden states.

        Returns:
            Tuple of (output word, new hidden state).
        """
        attention = SelfAttention(s_prev.shape[1])
        context_vector, attention_weights = attention(s_prev, hidden_states)
        x_embedded = self.embedding(x)

        # Expand context vector to match the dimensions for concatenation
        context_vector_expanded = tf.expand_dims(context_vector, axis=1)
        x_combined = tf.concat([context_vector_expanded, x_embedded], axis=-1)

        # Process the combined input through the GRU
        output_sequence, new_hidden_state = self.gru(x_combined)

        # Reshape and apply the Dense layer to get the output word
        output_word = tf.reshape(output_sequence, (-1,
            output_sequence.shape[2]))
        output_word = self.F(output_word)

        return output_word, new_hidden_state
