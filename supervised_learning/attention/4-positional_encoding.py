#!/usr/bin/env python3
"""
Calculate the positional encoding for a transformer.
"""

import numpy as np


def get_angle(pos, i, dm):
    """
    Calculates the angle for positional encoding.

    Args:
        pos (int): The position index.
        i (int): The index for the dimension.
        dm (int): Model depth.

    Returns:
        float: Calculated angle for the given position and index.
    """
    return pos / (10000 ** (i / dm))


def positional_encoding(max_seq_len, dm):
    """
    Calculates the positional encoding for a transformer.

    Args:
        max_seq_len (int): Maximum sequence length.
        dm (int): Model depth.

    Returns:
        np.ndarray: Positional encoding vectors of shape (max_seq_len, dm).
    """
    positional_enc = np.zeros((max_seq_len, dm))

    for pos in range(max_seq_len):
        angles = get_angle(pos, np.arange(dm), dm)
        positional_enc[pos, 0::2] = np.sin(angles[0::2])   # sine for even indices
        positional_enc[pos, 1::2] = np.cos(angles[1::2])   # cosine for odd indices

    return positional_enc
