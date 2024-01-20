#!/usr/bin/env python3

def add_matrices2D(mat1, mat2):
    if len(mat1) != len(mat2):
        return None

    if isinstance(mat1, list) and isinstance(mat2, list):
        final_result = []
        for i in range(len (mat1)):
            result = [a + b for a, b in zip(mat1[i], mat2[i])]
            final_result.append(result)

    return final_result
