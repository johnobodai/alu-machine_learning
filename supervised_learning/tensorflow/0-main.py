#!/usr/bin/env python3

import tensorflow as tf
from 0-create_placeholders import create_placeholders

x, y = create_placeholders(784, 10)
print(x)
print(y)
