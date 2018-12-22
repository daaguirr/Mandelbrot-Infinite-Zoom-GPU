import numpy as np
from numba import cuda

N = 3
NN = 4
BITS = 32
w = np.uint64(1 << 32)
ones = np.uint32(w - 1)
T = 256
x0 = -0.7600189058857209
y0 = -0.0799516080512771
BLOCK_SIZE = 32
log2T = int(np.log2(T) + 0.5)


@cuda.jit(device=True)
def fill_zeros(a):
    for i in range(N + 1):
        a[i] = 0


@cuda.jit(device=True)
def copy(source, target):
    for i in range(N + 1):
        target[i] = source[i]
