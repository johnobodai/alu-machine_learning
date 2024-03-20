Bayesian Probability Calculations

This repository contains Python scripts for conducting Bayesian probability calculations related to a study on a revolutionary cancer drug. The scripts implement functions to calculate likelihood, intersection, marginal probability, and posterior probability.
Contents

    Description
    Usage
    Files
    Requirements
    License

Description

Bayesian probability is a statistical technique that applies probabilities to statistical problems. In this context, the scripts in this repository facilitate Bayesian probability calculations for a study on the probability of patients developing severe side effects from a cancer drug.

The scripts implement the following functions:

    likelihood(x, n, P): Calculates the likelihood of obtaining data on severe side effects given various hypothetical probabilities.
    intersection(x, n, P, Pr): Calculates the intersection of obtaining data with various hypothetical probabilities, considering prior beliefs.
    marginal(x, n, P, Pr): Calculates the marginal probability of obtaining the data.
    posterior(x, n, P, Pr): Calculates the posterior probability for the various hypothetical probabilities given the data.

Usage

To use these scripts, follow these steps:

    Clone the repository to your local machine.
    Ensure you have Python installed (version 3.6 or above).
    Install the required libraries using pip: pip install -r requirements.txt.
    Run the desired script using Python.

For example, to run the likelihood calculation script:

bash

python 0-main.py

Files

    0-likelihood.py: Script containing the implementation of the likelihood function.
    1-intersection.py: Script containing the implementation of the intersection function.
    2-marginal.py: Script containing the implementation of the marginal probability function.
    3-posterior.py: Script containing the implementation of the posterior probability function.
    0-main.py, 1-main.py, 2-main.py, 3-main.py: Example scripts demonstrating the usage of the respective functions.

Requirements

The scripts require Python 3.x and the following libraries:

    NumPy

License

This project is licensed under the MIT License - see the LICENSE file for details.
