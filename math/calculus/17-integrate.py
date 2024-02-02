def poly_integral(poly, C=0):
    # Check if poly is a list and C is an integer
    if not isinstance(poly, list) or not all(isinstance(coeff, (int, float)) for coeff in poly) or not isinstance(C, int):
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

    return result
