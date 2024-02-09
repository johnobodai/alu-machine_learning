#!/usr/bin/env python3
"""Normal distribution"""


class Normal:
    """
    Represents a Normal distribution.
    
    Attributes:
        mean (float): The mean of the distribution.
        stddev (float): The standard deviation of the distribution.
    """

    def __init__(self, data=None, mean=0., stddev=1.):
        """
        Initializes the Normal distribution.

        Args:
            data (list): A list of data to estimate the distribution.
                         Defaults to None.
            mean (float): The mean of the distribution. Defaults to 0.0.
            stddev (float): The standard deviation of the distribution.
                            Defaults to 1.0.

        Raises:
            ValueError: If stddev is not positive.
            TypeError: If data is not a list or contains less
                       than two data points.
        """
        self.mean = float(mean)
        self.stddev = float(stddev)
        if data is None:
            if self.stddev <= 0:
                raise ValueError("stddev must be a positive value")
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            self.mean = sum(data) / len(data)
            self.stddev = (
                sum([(x - self.mean) ** 2 for x in data]) / len(data)
            ) ** 0.5

    def z_score(self, x):
        """
        Calculates the z-score of a given x-value.

        Args:
            x (float): The x-value.

        Returns:
            float: The z-score of x.
        """
        return (x - self.mean) / self.stddev

    def x_value(self, z):
        """
        Calculates the x-value of a given z-score.

        Args:
            z (float): The z-score.

        Returns:
            float: The x-value of z.
        """
        return self.mean + z * self.stddev

    def pdf(self, x):
        """
        Calculates the value of the PDF for a given x-value.

        Args:
            x (float): The x-value.

        Returns:
            float: The PDF value for x.
        """
        return (
            2
            / (3.1415926536 ** 0.5)
            * (x - (x ** 3) / 3 + (x ** 5) / 10 - (x ** 7) / 42 + (x ** 9) / 216)
        )

    def cdf(self, x):
        """
        Calculates the value of the CDF for a given x-value.

        Args:
            x (float): The x-value.

        Returns:
            float: The CDF value for x.
        """
        return (
            1
            - 2.7182818285 ** (
                -((x - self.mean) ** 2) / (2 * self.stddev ** 2)
            )
        )

