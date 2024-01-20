#!/usr/bin/env python3
"""A function that returns the transpose of a 2D matrix"""

def matrix_transpose(matrix):
    """
    Returns the transpose of a 2D matrix.

    This function takes a 2D matrix as input and returns its transpose, where
    rows become columns and columns become rows.

    Parameters:
    - matrix (list): The input matrix for which the transpose is to be
                    calculated.

    Returns:
    - list: The transpose of the input matrix.
    """
    rows, columns = len(matrix), len(matrix[0])
    transposed_matrix = [[0] * rows for x in range(columns)]

    for i in range(rows):
        for j in range(columns):
            transposed_matrix[j][i] = matrix[i][j]

    return transposed_matrix
