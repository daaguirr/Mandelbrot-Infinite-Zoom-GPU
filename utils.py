import numpy as np
from numba import cuda
import numba

N = 3
NN = 4
BITS = 32
w = np.uint64(1 << 32)
ones = np.uint32(w - 1)


@cuda.jit(device=True)
def fill_zeros(a):
    for i in range(N + 1):
        a[i] = 0


@cuda.jit(device=True)
def copy(source, target):
    for i in range(N + 1):
        target[i] = source[i]
