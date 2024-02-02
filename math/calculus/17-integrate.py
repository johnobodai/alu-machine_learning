#!/usr/bin/env python3
'''Calculates the integral of a polynomial'''


def poly_integral(poly, C=0):
    """
    Calculate the integral of a polynomial.

    Args:
        poly (list): Coefficients representing a polynomial.
        C (int): Integration constant.

    Returns:
        list or None: New coefficients representing the integral of the polynomial.
                      Returns None if poly or C are not valid.
    """
    # Check if poly is a list and C is an integer
    is_valid_poly = isinstance(poly, list)
    is_valid_poly = is_valid_poly and all(isinstance(c, (int, float)) for c in poly)
    is_valid_poly = is_valid_poly and isinstance(C, int)

    if not is_valid_poly:
        return None

    # Initialize the result list with the integration constant C
    result = [C]

    # Iterate through the coefficients of the polynomial
    for i, coeff in enumerate(poly):
        # Check if the coefficient is a valid number
        if not isinstance(coeff, (int, float)):
            return None

        # Calculate the new coefficient after integration
        new_coeff = coeff / (i + 1)

        # If the result is a whole number, represent it as an integer
        if new_coeff.is_integer():
            new_coeff = int(new_coeff)

        # Append the new coefficient to the result list
        result.append(new_coeff)

    # Remove trailing zeros
    while result and result[-1] == 0:
        result.pop()

    return result
