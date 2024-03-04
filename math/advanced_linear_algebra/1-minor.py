#!/usr/bin/env python3
"""Calculates the minor matrix of a matrix"""


def minor(matrix):
    """
    Calculate the minor matrix of a matrix.

    Parameters:
        matrix (list of lists): The input matrix for which the minor
                                matrix is to be calculated.

    Returns:
        list of lists: The minor matrix of the input matrix.

    Raises:
        TypeError: If matrix is not a list of lists.
        ValueError: If matrix is not a square matrix or is empty.
    """
    if not isinstance(matrix, list) or not all(isinstance(row, list)\
            for row in matrix):
        raise TypeError("matrix must be a list of lists")

    if not matrix or not all(len(row) == len(matrix) for row in matrix):
        raise ValueError("matrix must be a non-empty square matrix")

    def determinant(sub_matrix):
        """
        Calculate the determinant of a submatrix.

        Parameters:
            sub_matrix (list of lists): The submatrix for which the
                        determinant is to be calculated.

        Returns:
            float: The determinant of the submatrix.
        """
        if len(sub_matrix) == 1:
            return sub_matrix[0][0]
        elif len(sub_matrix) == 2:
            return sub_matrix[0][0] * sub_matrix[1][1] -\
                sub_matrix[0][1] * sub_matrix[1][0]
        else:
            det = 0
            for i in range(len(sub_matrix)):
                sign = (-1) ** i
                cofactor = determinant([row[:i] + row[i+1:]\
                        for row in sub_matrix[1:]])
                det += sign * sub_matrix[0][i] * cofactor
            return det

    minor_mat = []
    for i in range(len(matrix)):
        row = []
        for j in range(len(matrix[0])):
            sub_matrix = [row[:j] + row[j+1:]\
    for row in (matrix[:i] + matrix[i+1:])]
            row.append(determinant(sub_matrix))
        minor_mat.append(row)

    return minor_mat
