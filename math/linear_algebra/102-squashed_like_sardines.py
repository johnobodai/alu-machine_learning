#!/usr/bin/env python3

def cat_matrices(mat1, mat2, axis=0):
    """
    Concatenates two matrices along a specific axis.

    Args:
    - mat1 (list): First matrix.
    - mat2 (list): Second matrix.
    - axis (int, optional): Axis along which to concatenate. Default is 0.

    Returns:
    - list: Resulting matrix after concatenation.
            Returns None if the matrices cannot be concatenated.
    """
    # Ensure the matrices have the same shape along the specified axis
    if axis == 0 and len(mat1[0]) != len(mat2[0]):
        return None
    elif axis == 1 and len(mat1) != len(mat2):
        return None

    # Perform concatenation along the specified axis
    if axis == 0:
        result = mat1 + mat2
    else:
        result = [row1 + row2 for row1, row2 in zip(mat1, mat2)]

    return result
