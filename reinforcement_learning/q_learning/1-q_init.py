#!/usr/bin/env python3
"""
This module contains the function q_init for initializing
the Q-table for a FrozenLake environment.
"""

import numpy as np


def q_init(env):
    """
    Initializes the Q-table as a numpy array of zeros.

    Args:
        env (gym.Env): The FrozenLakeEnv instance.

    Returns:
        numpy.ndarray: The initialized Q-table with zeros.
        Shape is (number of states, number of actions).
    """
    num_states = env.observation_space.n
    num_actions = env.action_space.n
    return np.zeros((num_states, num_actions))

