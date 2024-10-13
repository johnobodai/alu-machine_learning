#!/usr/bin/env python3
"""
Defines a class that inherits from tensorflow.keras.layers.Layer
to create a decoder block for a transformer
"""


import tensorflow as tf
MultiHeadAttention = __import__('6-multihead_attention').MultiHeadAttention


class DecoderBlock(tf.keras.layers.Layer):
    """
    Class to create a decoder block for a transformer

    class constructor:
        def __init__(self, dm, h, hidden, drop_rate=0.1)

    public instance attributes:
        mha1: first MultiHeadAttention layer
        mha2: second MultiHeadAttention layer
        dense_hidden: hidden dense layer with hidden units and relu activation
        dense_output: output dense layer with dm units
        layernorm1: first layer norm layer, with epsilon=1e-6
        layernorm2: second layer norm layer, with epsilon=1e-6
        layernorm3: third layer norm layer, with epsilon=1e-6
        dropout1: first dropout layer
        dropout2: second dropout layer
        dropout3: third dropout layer

    public instance method:
        call(self, x, encoder_output, training, look_ahead_mask, padding_mask):
            calls the decoder block and returns the block's output
    """

    def __init__(self, dm, h, hidden, drop_rate=0.1):
        """
        Class constructor

        parameters:
            dm [int]: represents the dimensionality of the model
            h [int]: represents the number of heads
            hidden [int]: number of hidden units in fully connected layer
            drop_rate [float]: the dropout rate

        sets the public instance attributes:
            mha1: first MultiHeadAttention layer
            mha2: second MultiHeadAttention layer
            dense_hidden: hidden dense layer with hidden units, relu activation
            dense_output: output dense layer with dm units
            layernorm1: first layer norm layer, with epsilon=1e-6
            layernorm2: second layer norm layer, with epsilon=1e-6
            layernorm3: third layer norm layer, with epsilon=1e-6
            dropout1: first dropout layer
            dropout2: second dropout layer
            dropout3: third dropout layer
        """
        super(DecoderBlock, self).__init__()
        self.mha1 = MultiHeadAttention(dm, h)
        self.mha2 = MultiHeadAttention(dm, h)
        self.dense_hidden = tf.keras.layers.Dense(units=hidden,
                                                  activation='relu')
        self.dense_output = tf.keras.layers.Dense(units=dm)
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(drop_rate)
        self.dropout2 = tf.keras.layers.Dropout(drop_rate)
        self.dropout3 = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, encoder_output, training, look_ahead_mask, padding_mask):
        """
        Calls the decoder block and returns the block's output

        parameters:
            x [tensor of shape (batch, target_seq_len, dm)]:
                contains the input to the decoder block
            encoder_output [tensor of shape (batch, input_seq_len, dm)]:
                contains the output of the encoder
            training [boolean]:
                determines if the model is in training
            look_ahead_mask:
                mask for the first multi-head attention layer
            padding_mask:
                mask for the second multi-head attention layer

        returns:
            [tensor of shape (batch, target_seq_len, dm)]:
                contains the block's output
        """
        attn1, _ = self.mha1(x, x, x, look_ahead_mask)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(x + attn1)

        attn2, _ = self.mha2(out1, encoder_output, encoder_output, padding_mask)
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(out1 + attn2)

        ffn_output = self.dense_hidden(out2)
        ffn_output = self.dense_output(ffn_output)
        ffn_output = self.dropout3(ffn_output, training=training)
        output = self.layernorm3(out2 + ffn_output)

        return output
