#!/usr/bin/env python3
"""A function that adds two 2D matrices element-wise."""

def add_matrices2D(mat1, mat2):
    """
    Adds two 2D matrices element-wise.

    Parameters:
    - mat1 (list): The first input matrix.
    - mat2 (list): The second input matrix.

    Returns:
    - list or None: The result of adding mat1 and mat2 element-wise,
                    or None if mat1 and mat2 have different lengths.
    """
    if len(mat1) != len(mat2):
        return None

    if isinstance(mat1, list) and isinstance(mat2, list):
        final_result = []
        for i in range(len(mat1)):
            result = [a + b for a, b in zip(mat1[i], mat2[i])]
            final_result.append(result)

    return final_result
