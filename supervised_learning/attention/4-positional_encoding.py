#!/usr/bin/env python3
"""
Defines a function that calculates the positional encoding for a transformer.
"""

import numpy as np


def compute_angle(position, index, depth):
    """
    Calculates the angles for the positional encoding formulas.

    Args:
        position (int): The position in the sequence.
        index (int): The index of the dimension.
        depth (int): The model depth.

    Returns:
        float: The computed angle for
     """
    rate = 1 / (10000 ** (index / depth))
    return position * rate


def positional_encoding(max_length, depth):
    """
    Args:
        position (int): The position in the sequence.
        index (int): The index of the dimension.
        depth (int): The model depth.

    Returns:
        float: The computed angle for
    """
    encoding_matrix = np.zeros((max_length, depth))

    for pos in range(max_length):
        for idx in range(0, depth, 2):
            # Sin for even indices of encoding_matrix
            encoding_matrix[pos, idx] = np.sin(compute_angle(pos, idx, depth))
            # Cos for odd indices of encoding_matrix
            encoding_matrix[pos, idx + 1] = em
            em = np.cos(compute_angle(pos, idx, depth))

    return encoding_matrix

