#!/usr/bin/env python3
"""
Main file for testing the q_init function.
"""

load_frozen_lake = __import__('0-load_env').load_frozen_lake
q_init = __import__('1-q_init').q_init

# Test case 1: Default environment
env = load_frozen_lake()
Q = q_init(env)
print(Q.shape)  # Output: (64, 4)

# Test case 2: Slippery environment
env = load_frozen_lake(is_slippery=True)
Q = q_init(env)
print(Q.shape)  # Output: (64, 4)

# Test case 3: Custom map description
desc = [['S', 'F', 'F'], ['F', 'H', 'H'], ['F', 'F', 'G']]
env = load_frozen_lake(desc=desc)
Q = q_init(env)
print(Q.shape)  # Output: (9, 4)

# Test case 4: Pre-made map
env = load_frozen_lake(map_name='4x4')
Q = q_init(env)
print(Q.shape)  # Output: (16, 4)

