#!/usr/bin/env python3
"""
Main file for testing the load_frozen_lake function.
"""

load_frozen_lake = __import__('0-load_env').load_frozen_lake
import numpy as np

np.random.seed(0)

# Test case 1: Default environment
env = load_frozen_lake()
print(env.desc)
print(env.P[0][0])

# Test case 2: Slippery environment
env = load_frozen_lake(is_slippery=True)
print(env.desc)
print(env.P[0][0])

# Test case 3: Custom map description
desc = [['S', 'F', 'F'], ['F', 'H', 'H'], ['F', 'F', 'G']]
env = load_frozen_lake(desc=desc)
print(env.desc)

# Test case 4: Pre-made map
env = load_frozen_lake(map_name='4x4')
print(env.desc)

