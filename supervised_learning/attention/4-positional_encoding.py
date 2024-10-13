#!/usr/bin/env python3
"""
Calculates the positional encoding for a transformer.
"""

import numpy as np


def compute_angle(position, index, depth):
    """
    Calculates the angles for the positional encoding formulas.

    PE(position, 2i) = sin(position / 10000^(2i / depth))
    PE(position, 2i + 1) = cos(position / 10000^(2i / depth))
    """
    rate = 1 / (10000 ** (index / depth))
    return position * rate


def positional_encoding(max_length, depth):
    """
    Calculates the positional encoding for a transformer.

    parameters:
        max_length [int]:
            Represents the maximum sequence length.
        depth [int]:
            Model depth.

    returns:
        [numpy.ndarray of shape (max_length, depth)]:
            Contains the positional encoding vectors.
    """
    encoding_matrix = np.zeros((max_length, depth))

    for pos in range(max_length):
        for idx in range(0, depth, 2):
            # Sin for even indices of encoding_matrix
            encoding_matrix[pos, idx] = np.sin(compute_angle(pos, idx, depth))
            # Cos for odd indices of encoding_matrix
            encoding_matrix[pos, idx + 1] = np.cos(compute_angle(pos, idx, depth))

    return encoding_matrix

