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
    # Initialize the positional encoding matrix
    pos_enc = np.zeros((max_seq_len, dm))

    # Calculate the positional encoding values
    for pos in range(max_seq_len):
        for i in range(0, dm, 2):
            pos_enc[pos, i] = np.sin(pos / (10000 ** (i / dm)))
            if i + 1 < dm:
                pos_enc[pos, i + 1] = np.cos(
                    pos / (10000 ** ((i + 1) / dm))
                )

    return pos_enc
