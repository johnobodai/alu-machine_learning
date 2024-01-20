#!/usr/bin/env python3
"""A function that adds two arrays element-wise."""


def add_arrays(arr1, arr2):
    """
    Adds two arrays element-wise.

    Parameters:
    - arr1 (list): The first input array.
    - arr2 (list): The second input array.

    Returns:
    - list or None: The result of adding arr1 and arr2 element-wise,
                    or None if arr1 and arr2 are not the same shape.
    """
    if not (isinstance(arr1, list) and isinstance(arr2, list)):
        return None

    if len(arr1) != len(arr2):
        return None

    result = [a + b for a, b in zip(arr1, arr2)]

    return result
