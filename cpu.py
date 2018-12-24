import numpy as np
from numba import jit

BITS = 32
w = np.uint64(1 << 32)
ones = np.uint32(w - 1)


@jit(nopython=True)
def encode(x, n):
    sign_bit = np.uint32(x < 0)
    x = abs(x)

    ans = np.empty(n + 1, dtype=np.uint32)
    for i in range(n):
        ans[i] = np.uint32(x)
        x = (x - np.uint32(x)) * w

    ans[n] = sign_bit
    return ans


def naive_encode(x, n):
    sign_bit = np.uint32(x < 0)
    x = abs(x)

    ans = np.empty(n + 1, dtype=np.uint32)
    for i in range(n):
        ans[i] = np.uint32(x)
        x = (x - np.uint32(x)) * w

    ans[n] = sign_bit
    return ans


w1 = 2. ** -BITS
w2 = 2. ** (-2 * BITS)


@jit(nopython=True)
def decode(a):
    n = len(a) - 1
    x = a[0] + a[1] * w1 + a[2] * w2
    if a[n]:
        x = -x
    return x


@jit(nopython=True)
def uadd(a, b, ans):
    n = len(a) - 1
    carry = np.uint64(0)
    for i in range(n, -1, -1):
        abc = np.uint64(a[i]) + np.uint64(b[i]) + carry
        ans[i] = abc
        carry = np.uint64(abc >> BITS)


@jit(nopython=True)
def usub(a, b, ans):
    n = len(a) - 1
    aux = b.copy()
    i = n - 1
    while i >= 0 and b[i] == 0:
        aux[i] = ones
        i -= 1
    if i >= 0:
        aux[i] -= 1
    for j in range(n):
        aux[j] ^= ones
    uadd(a, aux, ans)


@jit(nopython=True)
def ucompare(a, b):
    n = len(a) - 1
    for i in range(n):
        if a[i] != b[i]:
            return np.int64(a[i]) - np.int64(b[i])
    return np.int64(0)


@jit(nopython=True)
def compare(a, b):
    n = len(a) - 1
    if a[n] != b[n]:
        return np.int64(b[n]) - np.int64(a[n])
    return ucompare(a, b)


@jit(nopython=True)
def lsh32(a, b):
    n = len(a) - 1
    for i in range(n):
        if i <= n - 1 - b:
            a[i] = a[i + b]
        else:
            a[i] = 0


@jit(nopython=True)
def rsh32(a, b):
    n = len(a) - 1
    for i in range(n, -1, -1):
        if i - b >= 0:
            a[i] = a[i - b]
        else:
            a[i] = 0


@jit(nopython=True)
def umuli(a, b, ans):
    n = len(a) - 1
    b = np.uint64(b)
    carry = np.uint64(0)
    for i in range(n, -1, -1):
        abc = np.uint64(a[i]) * b + carry
        ans[i] = abc
        carry = np.uint64(abc >> BITS)
    ans[n] = a[n]


@jit(nopython=True)
def mul(a, b, ans):
    n = len(a) - 1
    aux = a.copy()
    tmp = np.zeros_like(a)
    tmp2 = np.zeros_like(a)
    for i in range(n):
        umuli(aux, b[i], tmp)
        uadd(ans, tmp, tmp2)
        for j in range(n):
            ans[j] = tmp2[j]
        rsh32(aux, 1)
    ans[n] = a[n] ^ b[n]


@jit(nopython=True)
def add(a, b, ans):
    n = len(a) - 1
    if a[n] == b[n]:
        uadd(a, b, ans)
        ans[n] = a[n]
    elif ucompare(a, b) >= 0:
        usub(a, b, ans)
        ans[n] = a[n]
    else:
        usub(b, a, ans)
        ans[n] = b[n]


# arithmetic shift
@jit(nopython=True)
def rsh(a, b, ans):
    n = len(a) - 1
    k = b % BITS
    for i in range(n, -1, -1):
        l = i - (b + BITS - 1) // BITS
        r = i - b // BITS
        al = a[l] if l >= 0 else 0
        ar = a[r] if r >= 0 else 0
        sl = (al << (32 - k)) & ones
        sr = (ar >> k) & ones
        ans[i] = sl | sr
    ans[n] = a[n]


@jit(nopython=True)
def sub(a, b, ans):
    n = len(a) - 1
    if a[n] != b[n]:
        uadd(a, b, ans)
        ans[n] = a[n]
    elif ucompare(a, b) >= 0:
        usub(a, b, ans)
        ans[n] = a[n]
    else:
        usub(b, a, ans)
        ans[n] = a[n] ^ 1


def main():
    pi = np.array([3, 608135816, 2242054355, 0], dtype=np.uint32)
    minus_pi = np.array([3, 608135816, 2242054355, 1], dtype=np.uint32)
    three = np.array([3, 0, 0, 0], dtype=np.uint32)
    print(pi)
    ans = np.zeros_like(pi, dtype=np.uint32)
    sub(minus_pi, three, ans)
    print(ans)
    print(decode(ans))


if __name__ == '__main__':
    main()
