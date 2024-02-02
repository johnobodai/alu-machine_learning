#!/usr/bin/env python3
'''Calculate the derivative of a polynomial '''


def poly_derivative(poly):
    """
    Calculate the derivative of a polynomial.

    Parameters:
    - poly (list): Coefficients representing a polynomial.

    Returns:
    - list or None: New coefficients representing the derivative of the polynomial.
                    Returns None if poly is not valid.
    """
    if not isinstance(poly, list):
        return None

    if len(poly) < 2:
        return None

    result = []

    for i in range(1, len(poly)):
        if not isinstance(poly[i], (int, float)):
            return None

        new_coeff = i * poly[i]
        result.append(new_coeff)

    if not result:
        return [0]

    return result
