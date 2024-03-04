#!/usr/bin/env python3
def determinant(matrix):
    # Check if matrix is a list of lists
    if not isinstance(matrix, list) or not all(isinstance(row, list) for row in matrix):
        raise TypeError("matrix must be a list of lists")

    # Check if matrix is square
    num_rows = len(matrix)
    if num_rows == 0:
        return 0  # Empty matrix has determinant 0
    num_cols = len(matrix[0])
    if num_rows != num_cols:
        raise ValueError("matrix must be a square matrix")

    # Base case for 1x1 matrix
    if num_rows == 1:
        return matrix[0][0]

    # Recursive calculation of determinant using cofactor expansion along the first row
    det = 0
    for j in range(num_cols):
        sign = (-1) ** j
        submatrix = [row[:j] + row[j+1:] for row in matrix[1:]]  # Exclude jth column
        det += sign * matrix[0][j] * determinant(submatrix)

    return det

