#!/usr/bin/env python3
"""Binomial distribution"""


class Binomial:
    """
    Represents a binomial distribution.
    
    Attributes:
        n (int): The number of Bernoulli trials.
        p (float): The probability of a "success".
    """

    def __init__(self, data=None, n=1, p=0.5):
        """
        Initializes the Binomial distribution.

        Args:
            data (list): A list of data to estimate the distribution.
                         Defaults to None.
            n (int): The number of Bernoulli trials. Defaults to 1.
            p (float): The probability of a "success". Defaults to 0.5.

        Raises:
            ValueError: If n is not a positive value or p is not a valid probability.
            TypeError: If data is not a list or contains less than two data points.
        """
        if data is None:
            if n <= 0:
                raise ValueError("n must be a positive value")
            if p <= 0 or p >= 1:
                raise ValueError("p must be greater than 0 and less than 1")
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            p = sum(data) / len(data)
            n = round(sum(data) / p)
            p = sum(data) / n
        self.n = int(n)
        self.p = float(p)

    def pmf(self, k):
        """
        Calculates the value of the PMF for a given number of "successes".

        Args:
            k (int): The number of "successes".

        Returns:
            float: The PMF value for k.
        """
        k = int(k)
        if k < 0 or k > self.n:
            return 0
        return self.combination(self.n, k) * self.p**k * (1 - self.p)**(self.n - k)

    def cdf(self, k):
        """
        Calculates the value of the CDF for a given number of "successes".

        Args:
            k (int): The number of "successes".

        Returns:
            float: The CDF value for k.
        """
        k = int(k)
        if k < 0:
            return 0
        return sum(self.pmf(i) for i in range(k + 1))

    def combination(self, n, k):
        """
        Calculates the combination of n choose k.

        Args:
            n (int): Total number of items.
            k (int): Number of items to choose.

        Returns:
            int: The combination value.
        """
        if k == 0:
            return 1
        numerator = 1
        for i in range(n, n - k, -1):
            numerator *= i
        denominator = 1
        for i in range(1, k + 1):
            denominator *= i
        return numerator // denominator
