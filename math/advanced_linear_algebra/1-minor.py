#!/usr/bin/env python3
"""Calculates the minor matrix of a matrix"""


def calculate_minor_matrix(matrix):
    """
    Calculate the minor matrix of a matrix.

    Parameters:
        matrix (list of lists): The input matrix.

    Returns:
        list of lists: The minor matrix of the input matrix.
    """
    if not isinstance(matrix, list) or matrix == []:
        raise TypeError('matrix must be a list of lists')
    if any(not isinstance(row, list) for row in matrix):
        raise TypeError('matrix must be a list of lists')
    if any(len(row) != len(matrix) for row in matrix):
        raise ValueError('matrix must be a non-empty square matrix')

    if len(matrix) == 1:
        return [[1]]

    minor_matrix = []
    for x in range(len(matrix)):
        temp = []
        for y in range(len(matrix[0])):
            sub_matrix = []
            for row in (matrix[:x] + matrix[x + 1:]):
                sub_matrix.append(row[:y] + row[y + 1:])
            temp.append(calculate_determinant(sub_matrix))
        minor_matrix.append(temp)
    return minor_matrix
