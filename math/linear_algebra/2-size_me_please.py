#!/usr/bin/env python3
def matrix_shape(matrix):
    """
    Calculates the shape of a matrix

    Parameters:
    - matrix (list): The input matrix for which the shape is to be determined

    Returns:
    - list: A list representing the shape of the matrix, where each element 
            corresponds to the size of a dimension
    """
    shape = []
    while isinstance(matrix, list):
        shape.append(len(matrix))
        if matrix:
            matrix = matrix[0]
        else:
            break
    return shape
