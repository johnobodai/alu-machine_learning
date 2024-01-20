#!/usr/bin/env python3
"""A function that performs element-wise addition, subtraction, multiplication, and division of two numpy.ndarrays."""

def np_elementwise(mat1, mat2):
    """
    Performs element-wise addition, subtraction, multiplication, and division of two numpy.ndarrays.

    Parameters:
    - mat1 (numpy.ndarray): The first input numpy array.
    - mat2 (numpy.ndarray): The second input numpy array.

    Returns:
    - tuple: A tuple containing the element-wise sum, difference, product, and quotient, respectively.
    """
    return (mat1 + mat2, mat1 - mat2, mat1 * mat2, mat1 / mat2)
