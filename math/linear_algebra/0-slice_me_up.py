#!/usr/bin/env python3
import numpy as np

arr = np.array([9, 8, 2, 3, 9, 4, 1, 0, 3])
arr1, arr2, arr3 = arr[0:2], arr[-5:], arr[1:6]
print("The first two numbers of the array are: {}".format(arr1))
print("The last five numbers of the array are: {}".format(arr2))
print("The 2nd through 6th number of the array are: {}".format(arr3))
