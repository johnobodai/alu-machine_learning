#!/usr/bin/env python3
"""Poisson distrubution"""


class Poisson:
    """
    Represents a Poisson distribution.
    Attributes:
        lambtha (float): The expected number of occurrences
                         in a given time frame.
    """

    def __init__(self, data=None, lambtha=1.):
        """
        Initializes the Poisson distribution.

        Args:
            data (list): A list of data to estimate the distribution.
                         Defaults to None.
            lambtha (float): The expected number of occurrences.
                             Defaults to 1.0.

        Raises:
            ValueError: If lambtha is not positive.
            TypeError: If data is not a list or contains less
                       than two data points.
        """
        self.lambtha = float(lambtha)
        if data is None:
            if self.lambtha <= 0:
                raise ValueError(
                    "lambtha must be a positive value"
                )
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError(
                    "data must contain multiple values"
                )
            self.lambtha = sum(data) / len(data)

    def pmf(self, k):
        """
        Calculates the probability mass function (PMF)
        for a given number of successes.

        Args:
            k (int): The number of successes.

        Returns:
            float: The PMF value for k.

        Raises:
            ValueError: If k is not an integer.
        """
        k = int(k)
        if k < 0:
            return 0
        return (
            (self.lambtha ** k)
            * (2.71828 ** (-self.lambtha))
            / self.factorial(k)
        )

    def cdf(self, k):
        """
        Calculates the cumulative distribution function (CDF)
        for a given number of successes.

        Args:
            k (int): The number of successes.

        Returns:
            float: The CDF value for k.

        Raises:
            ValueError: If k is not an integer.
        """
        k = int(k)
        if k < 0:
            return 0
        cdf = 0
        for i in range(k + 1):
            cdf += self.pmf(i)
        return cdf

    def factorial(self, n):
        """
        Computes the factorial of a given number.

        Args:
            n (int): The number to compute the factorial for.

        Returns:
            int: The factorial value of n.
        """
        if n == 0:
            return 1
        result = 1
        for i in range(1, n + 1):
            result *= i
        return result
