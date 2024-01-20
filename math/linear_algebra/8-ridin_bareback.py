#!/usr/bin/env python3

def mat_mul(mat1, mat2):
    """
    Performs matrix multiplication.

    Args:
    - mat1 (list): First matrix (2D list).
    - mat2 (list): Second matrix (2D list).

    Returns:
    - list: Resulting matrix of the multiplication.
            Returns None if the matrices cannot be multiplied.
    """
    # Check if the matrices can be multiplied
    if len(mat1[0]) != len(mat2):
        return None

    # Initialize the result matrix with zeros
    result = [[0] * len(mat2[0]) for _ in range(len(mat1))]

    # Perform matrix multiplication
    for i in range(len(mat1)):
        for j in range(len(mat2[0])):
            for k in range(len(mat2)):
                result[i][j] += mat1[i][k] * mat2[k][j]

    return result

