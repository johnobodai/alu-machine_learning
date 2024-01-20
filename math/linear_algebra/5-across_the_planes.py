#!/usr/bin/env python3
"""A function that adds two 2D matrices element-wise."""

def add_matrices2D(mat1, mat2):
    """
    Adds two matrices element-wise.

    Parameters:
    - mat1 (list): The first input 2D matrix.
    - mat2 (list): The second input 2D matrix.

    Returns:
    - list or None: A new 2D matrix resulting from the element-wise addition,
                    or None if mat1 and mat2 are not the same shape.
    """
    # Check if the matrices have the same shape
    if len(mat1) != len(mat2) or any(len(row1) != len(row2) for row1, row2 in zip(mat1, mat2)):
        return None

    # Perform element-wise addition
    result = [[a + b for a, b in zip(row1, row2)] for row1, row2 in zip(mat1, mat2)]

    return result

