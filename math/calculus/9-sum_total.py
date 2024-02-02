#!/usr/bin/env python3
'''A function that calculates as Sigma notation'''


def summation_i_squared(n):

    '''
    Calculates the sigma squared of an input with a limit.

    Parameters:
    - n (int): The stopping condition for the sum.

    Returns:
    - int or None: The interger value of the sum. Returns None if
                   n is not a valid number
    '''
    if not isinstance(n, int) or n < 0:
        return None
    if n == 0:
        return 0
    else:
        return n ** 2 + summation_i_squared(n-1)
