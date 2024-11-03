#!/usr/bin/env python3
"""
Main file
"""
import gym
import numpy as np
from policy_gradient import policy_gradient

env = gym.make('CartPole-v1')
np.random.seed(1)

weight = np.random.rand(4, 2)
state, _ = env.reset()  # Unpack the tuple returned by reset
state = state[None, :]  # Reshape for compatibility

print(weight)
print(state)

action, grad = policy_gradient(state, weight)
print(action)
print(grad)

env.close()

