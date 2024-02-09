#!/usr/bin/env python3
"""Exponential distribution"""


class Exponential:
    """
    Represents an Exponential distribution.

    Attributes:
        lambtha (float): The expected number of occurrences
                         in a given time frame.
    """

    def __init__(self, data=None, lambtha=1.):
        """
        Initializes the Exponential distribution.

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
            self.lambtha = 1 / (sum(data) / len(data))

    def pdf(self, x):
        """
        Calculates the probability density function (PDF)
        for a given time period.

        Args:
            x (float): The time period.

        Returns:
            float: The PDF value for x.

        Raises:
            ValueError: If x is not a positive value.
        """
        if x < 0:
            return 0
        return self.lambtha * 2.7182818285 ** (-self.lambtha * x)

    def cdf(self, x):
        """
        Calculates the cumulative distribution function (CDF)
        for a given time period.

        Args:
            x (float): The time period.

        Returns:
            float: The CDF value for x.

        Raises:
            ValueError: If x is not a positive value.
        """
        if x < 0:
            return 0
        return 1 - 2.7182818285 ** (-self.lambtha * x)
