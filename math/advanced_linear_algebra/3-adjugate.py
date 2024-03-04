#!/usr/bin/env python3
"""Getting the adjugate matrix of a matrix"""


def adjugate(matrix):
    """Function that calculates the adjugate of a matrix"""
    cofactor_matrix = minor(matrix)
    transposed_matrix = []
    for x in range(len(cofactor_matrix)):
        transposed_matrix.append([])
        for y in range(len(cofactor_matrix)):
            transposed_matrix[x].append(cofactor_matrix[y][x])
    return transposed_matrix


def determinant(matrix):
    """Function that calculates the determinant of a matrix"""
    if len(matrix) == 1:
        return matrix[0][0]
    if len(matrix) == 2:
        det = ((matrix[0][0] * matrix[1][1])
                  - (matrix[0][1] * matrix[1][0]))
        return det

    det = 0
    for i, j in enumerate(matrix[0]):
        sub_matrix = [row[1:] for row in matrix[1:]]
        sub_matrix = [sub_matrix[k][:i] + sub_matrix[k][i+1:] for k in range(len(sub_matrix))]
        det += j * (-1) ** i * determinant(sub_matrix)
    return det


def minor(matrix):
    """Function that calculates the minor matrix of a matrix"""
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
        t = []
        for y in range(len(matrix[0])):
            s = []
            for row in (matrix[:x] + matrix[x + 1:]):
                s.append(row[:y] + row[y + 1:])
            sign = (-1) ** ((x + y) % 2)
            t.append(determinant(s) * sign)
        minor_matrix.append(t)
    return minor_matrix
