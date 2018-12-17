import numpy as np
import ctypes

BITS = 32


def iumul(a, b):
    ans = 0
    while b:
        if b & 1:
            ans += a
        a += a
        b >>= 1
    return ans


def decode(n):
    a = 0
    for i, x in enumerate(n):
        a += x * (2 ** (-i * 32))
    return a


def uadd(a, b, ans):
    return 1


def usub(a, b, ans):
    return -1


def compare(a, b):
    return 0


def umul(a, b, ans):
    return 0


def mul(a, b, ans):
    sgn_a = np.int32(a[0]) < 0
    sgn_b = np.int32(b[0]) < 0

    if sgn_a > 0 and sgn_b > 0:
        uadd(a, b, ans)
    elif sgn_a > 0 and sgn_b < 0:
        b[0] = -b[0]
        umul(a, b, ans)
        ans[0] = -ans[0]
        b[0] = -b[0]

    elif sgn_a < 0 and sgn_b > 0:
        a[0] = -a[0]
        umul(a, b, ans)
        ans[0] = -ans[0]
        a[0] = -a[0]

    elif sgn_a < 0 and sgn_b < 0:
        a[0] = -a[0]
        b[0] = -b[0]
        uadd(a, b, ans)
        a[0] = -a[0]
        b[0] = -b[0]


def add(a, b, ans):
    sgn_a = np.int32(a[0]) < 0
    sgn_b = np.int32(b[0]) < 0

    if sgn_a > 0 and sgn_b > 0:
        uadd(a, b, ans)
    elif sgn_a > 0 and sgn_b < 0:
        b[0] = -b[0]
        if compare(a, b) >= 0:
            usub(a, b, ans)
        else:
            usub(b, a, ans)
            ans[0] = -ans[0]

        b[0] = -b[0]

    elif sgn_a < 0 and sgn_b > 0:
        a[0] = -a[0]
        if compare(a, b) >= 0:
            usub(a, b, ans)
            ans[0] = -ans[0]

        else:
            usub(b, a, ans)
        a[0] = -a[0]

    elif sgn_a < 0 and sgn_b < 0:
        a[0] = -a[0]
        b[0] = -b[0]
        uadd(a, b, ans)
        ans[0] = -ans[0]
        a[0] = -a[0]
        b[0] = -b[0]


def main():
    pi = np.array([3, 608135816, 2242054355], dtype=np.uint32)


if __name__ == '__main__':
    main()
