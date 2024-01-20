#!/usr/bin/env python3
"""A function that adds two 2D matrices element-wise."""

def add_matrices2D(mat1, mat2):
    """
    Adds two matrices element-wise.

    Parameters:
    - mat1 (list): The first input 2D matrix.
    - mat2 (list): The second input 2D matrix.

    Returns:
    - list or None: New 2D matrix from element-wise addition,
                    or None if mat1 and mat2 are not the same shape.
    """
    # Check if the matrices have the same shape
    if len(mat1) != len(mat2):
        return None
    
    # Check if all rows have the same length
    for row1, row2 in zip(mat1, mat2):
        if len(row1) != len(row2):
            return None

    # Initialize an empty matrix for the result
    result = []
    
    # Perform element-wise addition using nested loops
    for row1, row2 in zip(mat1, mat2):
        result_row = []
        for a, b in zip(row1, row2):
            result_row.append(a + b)
        result.append(result_row)

    return result

