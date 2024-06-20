#!/usr/bin/env python3
import numpy as np

def sensitivity(confusion):
    """
    Calculates the sensitivity for each class in a confusion matrix.

    Args:
        confusion (np.ndarray): Confusion matrix of shape (classes, classes) where
                                row indices represent the correct labels and column
                                indices represent the predicted labels.

    Returns:
        np.ndarray: Array of shape (classes,) containing the sensitivity of each class.
    """
    # True positives are the diagonal elements
    true_positives = np.diag(confusion)

    # False negatives are the sum of the rows, minus the true positives
    false_negatives = np.sum(confusion, axis=1) - true_positives

    # Calculate sensitivity
    sensitivity_values = true_positives / (true_positives + false_negatives)

    return sensitivity_values
