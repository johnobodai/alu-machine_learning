#!/usr/bin/env python3
"""
Main file for testing the epsilon_greedy function.
"""

load_frozen_lake = __import__('0-load_env').load_frozen_lake
q_init = __import__('1-q_init').q_init
epsilon_greedy = __import__('2-epsilon_greedy').epsilon_greedy
import numpy as np

# Custom map description
desc = [['S', 'F', 'F'], ['F', 'H', 'H'], ['F', 'F', 'G']]
env = load_frozen_lake(desc=desc)
Q = q_init(env)

# Modify the Q-table for testing purposes
Q[7] = np.array([0.5, 0.7, 1, -1])

# Test epsilon-greedy function with different random seeds
np.random.seed(0)
print(epsilon_greedy(Q, 7, 0.5))  # Expected output: 2

np.random.seed(1)
print(epsilon_greedy(Q, 7, 0.5))  # Expected output: 0

