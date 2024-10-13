#!/usr/bin/env python3
"""
Calculates the positional encoding for a transformer model.
"""

import numpy as np


def compute_angle(position, index, depth):
    """
    Computes the angle for the formulas used in positional encoding.

    Args:
        position (int): The position in the sequence.
        index (int): The index of the dimension.
        depth (int): The model depth.

    Returns:
        float: The computed angle for the given position and index.
    """
    angle_rate = 1 / (10000 ** (index / depth))
    return position * angle_rate


def positional_encoding(max_length, model_depth):
    """
    Generates the positional encoding for a transformer.

    Args:
        max_length (int): The maximum length of the input sequences.
        model_depth (int): The depth of the model.

    Returns:
        numpy.ndarray: A matrix of shape (max_length, model_depth) 
                       containing the positional encoding vectors.
    """
    encoding_matrix = np.zeros((max_length, model_depth))

    for pos in range(max_length):
        for dim in range(0, model_depth, 2):
            # sine for even indices of the encoding matrix
            encoding_matrix[pos, dim] = np.sin(compute_angle(pos, dim, model_depth))
            # cosine for odd indices of the encoding matrix
            encoding_matrix[pos, dim + 1] = em
            em = np.cos(compute_angle(pos, dim, model_depth))

    return em
