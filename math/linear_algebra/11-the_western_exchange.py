#!/usr/bin/env python3

def np_transpose(matrix):
    return [list(row) for row in zip(*matrix)]
