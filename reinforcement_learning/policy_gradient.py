#!/usr/bin/env python3
"""
Module for computing policies and Monte-Carlo policy gradients.
"""

import numpy as np

def policy(matrix, weight):
    """
    Computes the policy using a matrix and weight.
    
    Args:
        matrix (np.ndarray): The state matrix of shape (1, n).
        weight (np.ndarray): The weight matrix of shape (n, m).
    
    Returns:
        np.ndarray: A probability distribution over actions of shape (1, m).
    """
    z = np.dot(matrix, weight)  # Linear combination
    exp = np.exp(z - np.max(z, axis=1, keepdims=True))  # Stabilize softmax
    return exp / np.sum(exp, axis=1, keepdims=True)  # Softmax computation


def policy_gradient(state, weight):
    """
    Computes the Monte Carlo policy gradient based on state and weight.
    
    Args:
        state (np.ndarray): Matrix representing the current observation 
                            of the environment, shape (1, n).
        weight (np.ndarray): Matrix of random weights, shape (n, m).
    
    Returns:
        tuple: The chosen action (int) and its gradient (np.ndarray).
    """
    probs = policy(state, weight)  # Compute policy probabilities
    action = np.random.choice(probs.shape[1], p=probs.ravel())  # Select action
    
    # Compute gradient
    d_softmax = probs.copy()
    d_softmax[0, action] -= 1
    grad = np.dot(state.T, d_softmax)
    
    return action, grad

