#!/usr/bin/env python3
"""Getting the inverse matrix of a matrix"""


def inverse(matrix):
    """Function that calculates the inverse of a matrix"""
    adjugate_matrix = adjugate(matrix)
    determinant_matrix = det(matrix)
    if determinant_matrix == 0:
        return None
    return [[y / determinant_matrix for y in x] for x in adjugate_matrix]


def adjugate(matrix):
    """Function that calculates the adjugate of a matrix"""
    cofactor_matrix = minor(matrix)
    transposed_matrix = []
    for x in range(len(cofactor_matrix)):
        transposed_matrix.append([])
        for y in range(len(cofactor_matrix)):
            transposed_matrix[x].append(cofactor_matrix[y][x])
    return transposed_matrix


def det(matrix):
    """Function that calculates the determinant of a matrix"""
    if len(matrix) == 1:
        return matrix[0][0]
    if len(matrix) == 2:
        determinant_value = (matrix[0][0] * matrix[1][1]) - (matrix[0][1] * matrix[1][0])
        return determinant_value
    determinant_result = 0
    for x, num in enumerate(matrix):
        temp = []
        P = matrix[0][x]
        for row in matrix[1:]:
            new = []
            for j in range(len(matrix)):
                if j != x:
                    new.append(row[j])
            temp.append(new)
        determinant_result += P * determinant(temp) * (-1) ** x
    return determinant_result


def determinant(matrix):
    """Function that calculates the determinant"""
    if len(matrix) == 1:
        return matrix[0][0]
    if len(matrix) == 2:
        determ = ((matrix[0][0] * matrix[1][1])
                  - (matrix[0][1] * matrix[1][0]))
        return determ

    determ = 0
    for i, j in enumerate(matrix[0]):
        row = [r for r in matrix[1:]]
        temp = []
        for r in row:
            a = []
            for c in range(len(matrix)):
                if c != i:
                    a.append(r[c])
            temp.append(a)
        determ += j * (-1) ** i * determinant(temp)
    return determ


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
