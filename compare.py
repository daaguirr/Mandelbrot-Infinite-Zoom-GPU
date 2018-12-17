import numpy as np


def compare(a, b):
    n = len(a)
    if a[0] != b[0]:
        return np.int32(a[0]) - np.int32(b[0])
    for i in range(1, n):
        if a[i] != b[i]:
            return a[i] - b[i]
    return 0
