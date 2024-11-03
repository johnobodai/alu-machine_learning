#!/usr/bin/env python3
"""
Script to train a DQN agent to play Atari's Breakout.
"""

import gym
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.optimizers import Adam
from rl.agents.dqn import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import EpsGreedyQPolicy


def build_model(input_shape, n_actions):
    """
    Builds a neural network model for the DQN agent.

    Parameters:
    - input_shape: Shape of the input for the model.
    - n_actions: Number of possible actions.

    Returns:
    - Compiled Keras model.
    """
    model = Sequential()
    model.add(Flatten(input_shape=(1,) + input_shape))
    model.add(Dense(24, activation="relu"))
    model.add(Dense(24, activation="relu"))
    model.add(Dense(n_actions, activation="linear"))
    return model


def train_agent(env_name="Breakout-v0", episodes=50000):
    """
    Trains a DQN agent to play Breakout.

    Parameters:
    - env_name: Name of the Gym environment.
    - episodes: Number of episodes for training.

    Saves:
    - Trained policy network as policy.h5.
    """
    # Initialize environment and parameters
    env = gym.make(env_name)
    input_shape = env.observation_space.shape
    n_actions = env.action_space.n

    # Build model and agent
    model = build_model(input_shape, n_actions)
    memory = SequentialMemory(limit=100000, window_length=1)
    policy = EpsGreedyQPolicy()
    dqn = DQNAgent(model=model, nb_actions=n_actions, memory=memory,
                   nb_steps_warmup=10000, target_model_update=1e-2,
                   policy=policy, gamma=0.99)
    dqn.compile(Adam(lr=1e-3), metrics=["mae"])

    # Train the agent
    dqn.fit(env, nb_steps=episodes, visualize=False, verbose=2)

    # Save the policy network
    dqn.save_weights("policy.h5", overwrite=True)

    # Close the environment
    env.close()


if __name__ == "__main__":
    train_agent()

