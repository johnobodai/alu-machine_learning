#!/usr/bin/env python3
"""
Performs multi head attention
"""

import tensorflow as tf

def sdp_attention(Q, K, V, mask=None):
    """
    Calculates the scaled dot product attention.

    Args:
        Q (tensor): Query matrix of shape (..., seq_len_q, dk).
        K (tensor): Key matrix of shape (..., seq_len_v, dk).
        V (tensor): Value matrix of shape (..., seq_len_v, dv).
        mask (tensor, optional): Mask tensor that can be broadcast into 
                                 (..., seq_len_q, seq_len_v). Defaults to None.

    Returns:
        tuple:
            output (tensor): Tensor containing the scaled dot product attention
                             of shape (..., seq_len_q, dv).
            weights (tensor): Tensor containing the attention weights of shape
                              (..., seq_len_q, seq_len_v).
    """
    # Calculate the dot product of Q and K
    dk = tf.cast(tf.shape(K)[-1], tf.float32)  # Get the depth of keys
    scaled_attention_logits = tf.matmul(Q, K, transpose_b=True) / tf.math.sqrt(dk)

    # Apply the mask, if provided
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)

    # Calculate the attention weights
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)

    # Calculate the output
    output = tf.matmul(attention_weights, V)

    return output, attention_weights
