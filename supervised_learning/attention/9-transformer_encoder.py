#!/usr/bin/env python3
"""
Defines a class that inherits from tensorflow.keras.layers.Layer
to create the encoder for a transformer
"""

import tensorflow as tf
positional_encoding = __import__('4-positional_encoding').positional_encoding
EncoderBlock = __import__('7-transformer_encoder_block').EncoderBlock


class Encoder(tf.keras.layers.Layer):
    """Class Encoder for a transformer"""

    def __init__(self, N, dm, h, hidden, input_vocab,
                 max_seq_len, drop_rate=0.1):
        """
        Class constructor

        Parameters:
            N [int]: number of blocks in the encoder
            dm [int]: dimensionality of the model
            h [int]: number of heads
            hidden [int]: number of hidden units in fully connected layer
            input_vocab [int]: size of the input vocabulary
            max_seq_len [int]: maximum sequence length possible
            drop_rate [float]: dropout rate

        Sets the following public instance attributes:
            N: number of blocks in the encoder
            dm: dimensionality of the model
            embedding: embedding layer for the inputs
            positional_encoding: numpy.ndarray of shape (max_seq_len, dm)
            blocks: list of length N containing EncoderBlock instances
            dropout: dropout layer for positional encodings

        Returns:
            None.
        """
        super(Encoder, self).__init__()
        self.N = N
        self.dm = dm
        self.embedding = tf.keras.layers.Embedding(input_vocab, dm)
        self.positional_encoding = positional_encoding(max_seq_len, dm)
        self.blocks = [EncoderBlock(dm, h, hidden, drop_rate) for _ in range(N)]
        self.dropout = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, training, mask):
        """
        Calls the encoder and returns the encoder output

        Parameters:
            x [tensor of shape (batch, input_seq_len)]:
                contains the input to the encoder
            training [boolean]: determines if the model is in training
            mask: mask to be applied for multi-head attention

        Returns:
            [tensor of shape (batch, input_seq_len, dm)]:
                contains the encoder output
        """
        seq_len = tf.shape(x)[1]
        input_enc = self.embedding(x)
        input_enc += self.positional_encoding[:seq_len, :]
        output = self.dropout(input_enc, training=training)

        for block in self.blocks:
            output = block(output, training, mask)

        return output

