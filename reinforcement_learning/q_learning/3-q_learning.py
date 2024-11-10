#!/usr/bin/env python3
"""
This module contains the train function that performs Q-learning on the FrozenLake environment.
"""

import numpy as np
import random

# Importing epsilon_greedy function
epsilon_greedy = __import__('2-epsilon_greedy').epsilon_greedy

def train(env, Q, episodes=5000, max_steps=100, alpha=0.1, gamma=0.99,
          epsilon=1, min_epsilon=0.1, epsilon_decay=0.05):
    """
    Performs Q-learning to train the agent in the given environment.

    Args:
        env: The FrozenLakeEnv instance (environment).
        Q: The Q-table (numpy.ndarray) containing state-action values.
        episodes (int): The number of episodes for training.
        max_steps (int): The maximum number of steps per episode.
        alpha (float): The learning rate.
        gamma (float): The discount factor.
        epsilon (float): The initial epsilon for epsilon-greedy.
        min_epsilon (float): The minimum value that epsilon should decay to.
        epsilon_decay (float): The rate at which epsilon decays.

    Returns:
        Q (numpy.ndarray): The updated Q-table after training.
        total_rewards (list): A list of the rewards obtained per episode.
    """
    total_rewards = []

    for episode in range(episodes):
        state = env.reset()  # Reset environment to start a new episode
        total_reward = 0  # Initialize total reward for the episode

        for step in range(max_steps):
            # Choose the action using epsilon-greedy
            action = epsilon_greedy(Q, state, epsilon)

            # Perform the action and observe the new state and reward
            next_state, reward, done, _ = env.step(action)

            # If the agent falls into a hole, reward is -1
            if done and reward == 0:
                reward = -1

            # Update Q-table using the Q-learning formula
            Q[state, action] = Q[state, action] + alpha * (
                reward + gamma * np.max(Q[next_state]) - Q[state, action])

            total_reward += reward
            state = next_state

            # If the agent reaches the goal (done is True), break from the loop
            if done:
                break

        total_rewards.append(total_reward)

        # Decay epsilon
        epsilon = max(min_epsilon, epsilon - epsilon_decay)

    return Q, total_rewards

