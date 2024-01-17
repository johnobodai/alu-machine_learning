#!/usr/bin/env python3
def matrix_transpose(matrix):
    """
    Returns the transpose of a 2D matrix

    Parameters:
    - list: The transpose of the input matrix
    """
    rows, columns = len(martix), len(martix[0])
    transposed_matrix = [[0] * row for x in range(columns)]

    for i in range(rows):
        for j in range(columns):
            transposed_matrix[j][i] = martrix[i][j]

    return transposed_matrix
