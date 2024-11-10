#!/usr/bin/env python3
"""
This module contains the play function that allows the trained agent to play an episode.
"""

import numpy as np


def play(env, Q, max_steps=100):
    """
    Plays an episode in the environment using the Q-table.

    Args:
        env: The FrozenLakeEnv instance (environment).
        Q: The Q-table (numpy.ndarray) containing state-action values.
        max_steps (int): The maximum number of steps in the episode.

    Returns:
        total_reward (float): The total reward for the episode.
    """
    state = env.reset()  # Reset environment to start the episode
    total_reward = 0

    for step in range(max_steps):
        # Display the current state of the environment
        env.render()

        # Choose the action with the highest Q-value (exploit)
        action = np.argmax(Q[state])

        # Perform the action and observe the next state and reward
        next_state, reward, done, _ = env.step(action)

        total_reward += reward
        state = next_state

        # If the agent reaches the goal (done is True), break the loop
        if done:
            env.render()  # Render final state
            break

    return total_reward

