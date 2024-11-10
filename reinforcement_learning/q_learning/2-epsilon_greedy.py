#!/usr/bin/env python3
"""
This module contains the epsilon_greedy function to decide the
next action based on the epsilon-greedy algorithm.
"""

import numpy as np


def epsilon_greedy(Q, state, epsilon):
    """
    Determines the next action using the epsilon-greedy algorithm.

    Args:
        Q (numpy.ndarray): The Q-table containing the state-action values.
        state (int): The current state.
        epsilon (float): The epsilon value for exploration vs exploitation.

    Returns:
        int: The next action to take, determined by epsilon-greedy.
    """
    # Generate a random number between 0 and 1
    p = np.random.uniform(0, 1)

    # Explore: pick a random action
    if p < epsilon:
        action = np.random.randint(0, Q.shape[1])
    # Exploit: pick the best action based on the Q-table
    else:
        action = np.argmax(Q[state])

    return action

