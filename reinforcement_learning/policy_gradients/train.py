#!/usr/bin/env python3
"""
Module for training a policy gradient model.
"""

import numpy as np
from policy_gradient import policy_gradient

def train(env, nb_episodes, alpha=0.000045, gamma=0.98, show_result=False):
    """
    Trains a policy gradient model on the given environment.
    
    Args:
        env (gym.Env): The initialized environment.
        nb_episodes (int): Number of episodes used for training.
        alpha (float): Learning rate.
        gamma (float): Discount factor.
        show_result (bool): If True, renders the environment every 1000 episodes.
    
    Returns:
        list: Scores (sum of rewards) for each episode.
    """
    weight = np.random.rand(env.observation_space.shape[0], env.action_space.n)
    scores = []  # To store scores for each episode

    for episode in range(nb_episodes):
        state, _ = env.reset()  # Unpack the tuple, get the state
        state = state[None, :]  # Reshape for compatibility
        episode_rewards = []
        gradients = []

        while True:
            action, grad = policy_gradient(state, weight)  # Get action & gradient
            next_state, reward, done, _, _ = env.step(action)  # Take action
            episode_rewards.append(reward)
            gradients.append(grad)
            state = next_state[None, :]  # Update state

            if done:
                break

        # Compute discounted rewards
        discounted_rewards = np.zeros_like(episode_rewards)
        cumulative_reward = 0
        for t in reversed(range(len(episode_rewards))):
            cumulative_reward = episode_rewards[t] + gamma * cumulative_reward
            discounted_rewards[t] = cumulative_reward

        # Update weights using gradients and rewards
        for grad, reward in zip(gradients, discounted_rewards):
            weight += alpha * grad * reward

        score = sum(episode_rewards)
        scores.append(score)

        # Print episode and score on the same line
        print(f"Episode: {episode + 1}/{nb_episodes} | Score: {score}", end="\r", flush=True)

        # Render the environment every 1000 episodes if show_result is True
        if show_result and (episode + 1) % 1000 == 0:
            env.render()

    print("\nTraining completed.")
    return scores

