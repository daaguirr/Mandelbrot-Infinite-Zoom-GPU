import numpy as np
from numba import cuda

N = 10
NN = 11
BITS = 32
w = np.uint64(1 << 32)
ones = np.uint32(w - 1)
T = 1024
x0 = -0.7600189058857350
y0 = -0.0799516080512771
#x0 = np.float128(-1.7400623825793399052208441670658256382966417204361718668798624184611829)
#y0 = np.float128(-0.0281753397792110489924115211443195096875390767429906085704013095958801)
BLOCK_SIZE = 64
log2T = int(np.log2(T) + 0.5)


@cuda.jit(device=True)
def fill_zeros(a):
    for i in range(N + 1):
        a[i] = 0


@cuda.jit(device=True)
def copy(source, target):
    for i in range(N + 1):
        target[i] = source[i]
