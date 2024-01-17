#!/usr/bin/env python3
def matrix_shape(matrix):
    shape = []
    while matrix:
        shape.append(len(matrix))
        try:
            matrix = matrix[0]
        except TypeError:
            break
    return shape
