#!/usr/bin/env python3
"""
Main file for testing the play function with the trained agent.
"""

load_frozen_lake = __import__('0-load_env').load_frozen_lake
q_init = __import__('1-q_init').q_init
train = __import__('3-q_learning').train
play = __import__('4-play').play
import numpy as np

# Initialize environment and Q-table
np.random.seed(0)
desc = [['S', 'F', 'F'], ['F', 'H', 'H'], ['F', 'F', 'G']]
env = load_frozen_lake(desc=desc)
Q = q_init(env)

# Train the agent using Q-learning
Q, total_rewards = train(env, Q)

# Test the play function by having the agent play an episode
print(play(env, Q))

