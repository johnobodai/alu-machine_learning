#!/usr/bin/env python3
"""A function that concatenates two 2D matrices along a specific axis."""

def cat_matrices2D(mat1, mat2, axis=0):
    """
    Concatenates two 2D matrices along a specific axis.

    Parameters:
    - mat1 (list): The first input 2D matrix.
    - mat2 (list): The second input 2D matrix.
    - axis (int): The axis along which to concatenate (0 for rows, 1 for columns).

    Returns:
    - list or None: The result of concatenating mat1 and mat2 along the specified axis,
                    or None if the matrices cannot be concatenated.
    """
    if axis == 0 and len(mat1[0]) != len(mat2[0]):
        return None
    elif axis == 1 and len(mat1) != len(mat2):
        return None

    if axis == 0:
        return mat1 + mat2
    elif axis == 1:
        result = [row1 + row2 for row1, row2 in zip(mat1, mat2)]
        return result
    else:
        return None
