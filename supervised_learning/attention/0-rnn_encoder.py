#!/usr/bin/env python3
import tensorflow as tf

class RNNEncoder(tf.keras.layers.Layer):
    """
    RNN Encoder using GRU for machine translation.
    """

    def __init__(self, vocab, embedding, units, batch):
        """
        Initialize the encoder.
        
        Args:
            vocab (int): Vocabulary size.
            embedding (int): Embedding vector size.
            units (int): Number of GRU units.
            batch (int): Batch size.
        """
        super(RNNEncoder, self).__init__()
        self.batch = batch
        self.units = units
        self.embedding = tf.keras.layers.Embedding(vocab, embedding)
        self.gru = tf.keras.layers.GRU(
            units, 
            return_sequences=True, 
            return_state=True, 
            recurrent_initializer='glorot_uniform'
        )

    def initialize_hidden_state(self):
        """
        Initialize hidden state.
        
        Returns:
            Tensor of zeros for the hidden state.
        """
        return tf.zeros((self.batch, self.units))

    def call(self, x, initial):
        """
        Run the encoder.
        
        Args:
            x (Tensor): Input tensor.
            initial (Tensor): Initial hidden state.
        
        Returns:
            Tuple of (outputs, last hidden state).
        """
        x = self.embedding(x)
        outputs, hidden = self.gru(x, initial_state=initial)
        return outputs, hidden
