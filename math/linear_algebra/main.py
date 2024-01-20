#!/usr/bin/env python3

add_arrays = __import__('4-line_up').add_arrays

arr1 = [1, 2, 3, 4]
arr2 = [5, 6, 7, 8]

# Adding arrays element-wise
result = add_arrays(arr1, arr2)

# Printing the result and original arrays
print(f"Result of adding arrays: {result}")
print(f"Original arr1: {arr1}")
print(f"Original arr2: {arr2}")

# Example with arrays of different lengths
print(add_arrays(arr1, [1, 2, 3]))
