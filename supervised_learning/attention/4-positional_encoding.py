#!/usr/bin/env python3
"""
Calculate the positional encoding for a transformer.
"""

import numpy as np


def positional_encoding(max_seq_len, dm):
    """
    Calculate the positional encoding.

    Args:
        max_seq_len (int): Maximum sequence length.
        dm (int): Model depth.

    Returns:
        np.ndarray: Positional encoding vectors of shape (max_seq_len, dm).
    """
    # Create a matrix of shape (max_seq_len, dm)
    position = np.arange(max_seq_len)[:, np.newaxis]  # Shape (max_seq_len, 1)
    depth = np.arange(dm)[np.newaxis, :]  # Shape (1, dm)

    # Calculate the positional encodings
    angle_rates = 1 / np.power(10000, (2 * (depth // 2)) / np.float32(dm))
    angle = position * angle_rates  # Shape (max_seq_len, dm)

    # Apply sine to even indices and cosine to odd indices
    pos_enc = np.zeros((max_seq_len, dm))
    pos_enc[:, 0::2] = np.sin(angle)  # Apply sine to even indices
    pos_enc[:, 1::2] = np.cos(angle)  # Apply cosine to odd indices
