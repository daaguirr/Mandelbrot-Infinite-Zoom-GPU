import numpy as np
from numba import cuda
import numba
import cpu
from utils import N, BITS, ones, fill_zeros, copy, NN


@cuda.jit(device=True)
def uadd(a, b, ans):
    carry = np.uint64(0)
    for i in range(N, -1, -1):
        abc = np.uint64(a[i]) + np.uint64(b[i]) + carry
        ans[i] = abc
        carry = np.uint64(abc >> BITS)


@cuda.jit(device=True)
def usub(a, b, ans):
    aux = cuda.local.array(NN, numba.uint32)
    copy(a, aux)
    i = N - 1
    while i >= 0 and b[i] == 0:
        aux[i] = ones
        i -= 1
    if i >= 0:
        aux[i] -= 1
    for j in range(N):
        aux[j] ^= ones
    uadd(a, aux, ans)


@cuda.jit(device=True)
def ucompare(a, b):
    for i in range(N):
        if a[i] != b[i]:
            return np.int64(a[i]) - np.int64(b[i])
    return np.int64(0)


@cuda.jit(device=True)
def compare(a, b):
    if a[N] != b[N]:
        return np.int64(b[N]) - np.int64(a[N])
    return ucompare(a, b)


# tested until here ^^^
# test pending from here vvv


@cuda.jit(device=True)
def lsh32(a, b):
    for i in range(N):
        if i <= N - 1 - b:
            a[i] = a[i + b]
        else:
            a[i] = 0


@cuda.jit(device=True)
def rsh32(a, b):
    for i in range(N, -1, -1):
        if i - b >= 0:
            a[i] = a[i - b]
        else:
            a[i] = 0


@cuda.jit(device=True)
def umuli(a, b, ans):
    b = np.uint64(b)
    carry = np.uint64(0)
    for i in range(N, -1, -1):
        abc = np.uint64(a[i]) * b + carry
        ans[i] = abc
        carry = np.uint64(abc >> BITS)
    ans[N] = a[N]


@cuda.jit(device=True)
def mul(a, b, ans):
    aux = cuda.local.array(NN, numba.uint32)
    tmp = cuda.local.array(NN, numba.uint32)
    tmp2 = cuda.local.array(NN, numba.uint32)

    copy(a, aux)
    fill_zeros(tmp)
    fill_zeros(tmp2)

    for i in range(N):
        umuli(aux, b[i], tmp)
        uadd(ans, tmp, tmp2)
        copy(tmp2, ans)
        rsh32(aux, 1)
    ans[N] = a[N] ^ b[N]


@cuda.jit(device=True)
def add(a, b, ans):
    if a[N] == b[N]:
        uadd(a, b, ans)
        ans[N] = a[N]
    elif ucompare(a, b) >= 0:
        usub(a, b, ans)
        ans[N] = a[N]
    else:
        usub(b, a, ans)
        ans[N] = b[N]


# arithmetic shift
@cuda.jit(device=True)
def rsh(a, b, ans):
    k = b % BITS
    for i in range(N, -1, -1):
        l = i - (b + BITS - 1) // BITS
        r = i - b // BITS
        al = a[l] if l >= 0 else 0
        ar = a[r] if r >= 0 else 0
        sl = (al << (32 - k)) & ones
        sr = (ar >> k) & ones
        ans[i] = sl | sr
    ans[N] = a[N]


@cuda.jit(device=True)
def sub(a, b, ans):
    if a[N] != b[N]:
        uadd(a, b, ans)
        ans[N] = a[N]
    elif ucompare(a, b) >= 0:
        usub(a, b, ans)
        ans[N] = a[N]
    else:
        usub(b, a, ans)
        ans[N] = a[N] ^ 1


@cuda.jit
def kernel(a, b, ans):
    mul(a, b, ans)


def main():
    pi = np.array([3, 608135816, 2242054355, 0], dtype=np.uint32)
    minus_pi = np.array([3, 608135816, 2242054355, 1], dtype=np.uint32)
    three = np.array([3, 0, 0, 0], dtype=np.uint32)
    ans = np.zeros_like(pi, dtype=np.uint32)

    blockdim = (1,)
    griddim = (1,)

    d_pi = cuda.to_device(pi)
    d_minus_pi = cuda.to_device(minus_pi)
    d_three = cuda.to_device(three)
    d_ans = cuda.to_device(ans)
    kernel[griddim, blockdim](d_three, d_minus_pi, d_ans)
    d_ans.to_host()

    print(ans)
    print(cpu.decode(ans))


if __name__ == '__main__':
    main()
