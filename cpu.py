import numpy as np

BITS = 32


def iumul(a, b):
    ans = 0
    while b:
        if b & 1:
            ans += a
        a += a
        b >>= 1
    return ans


def encode(x, n):
    was_negative = x < 0
    x = abs(x)

    ans = np.empty(n, dtype=np.uint32)
    for i in range(n):
        ans[i] = int(x)
        x = (x - int(x)) * 2 ** 32

    if was_negative:
        ans[0] = -ans[0]
    return ans


def decode(n):
    was_negative = np.int32(n[0]) < 0
    if was_negative:
        n[0] = -n[0]
    a = 0
    for i, x in enumerate(n):
        a += x * (2 ** (-i * 32))
    if was_negative:
        a = -a
        n[0] = -n[0]
    return a


def uadd(a, b, ans):
    n = len(a)
    carry = np.int64(0)
    for i in reversed(range(n)):
        abc = np.int64(a[i]) + np.int64(b[i]) + np.int64(carry)
        ans[i] = abc
        carry = abc >> 32


def usub(a, b, ans):
    n = len(a)
    aux = b.copy()
    i = n - 1
    while i >= 0 and b[i] == 0:
        aux[i] = -1
        i -= 1
    if i >= 0:
        aux[i] -= 1
    for j in range(n):
        aux[j] ^= 2 ** 32 - 1
    uadd(a, aux, ans)


def compare(a, b):
    n = len(a)
    if a[0] != b[0]:
        return np.int32(a[0]) - np.int32(b[0])
    for i in range(1, n):
        if a[i] != b[i]:
            return a[i] - b[i]
    return 0


def lsh32(a, n):
    size = len(a)
    for i in range(size):
        if i <= size - 1 - n:
            a[i] = a[i + n]
        else:
            a[i] = 0


def rsh32(a, n):
    size = len(a)
    for i in reversed(range(size)):
        if i - n >= 0:
            a[i] = a[i - n]
        else:
            a[i] = 0


def umuli(a, b, ans):
    n = len(a)
    carry = 0
    for i in reversed(range(n)):
        abc = int(a[i]) * b + carry
        ans[i] = abc
        carry = abc >> 32


def umul(a, b, ans):
    n = len(a)
    aux = a.copy()
    tmp = np.zeros_like(a)
    tmp2 = np.zeros_like(a)
    print(f'ans: {ans}')
    for i in range(n):
        umuli(aux, b[i], tmp)
        uadd(ans, tmp, tmp2)
        for j in range(n):
            ans[j] = tmp2[j]
        print('')
        print(f'aux: {aux}')
        print(f'* {b[i]} =')
        print(f'tmp: {tmp}')
        print('ans += tmp')
        print(f'ans: {ans}')
        rsh32(aux, 1)


def mul(a, b, ans):
    sgn_a = np.int32(a[0]) >= 0
    sgn_b = np.int32(b[0]) >= 0

    if sgn_a and sgn_b:
        umul(a, b, ans)

    elif sgn_a and not sgn_b:
        b[0] = -b[0]
        umul(a, b, ans)
        print(f'ans antes de -ans: {ans}')
        ans[0] = -ans[0]
        print(f'ans despues de -ans: {ans}')
        b[0] = -b[0]

    elif not sgn_a and sgn_b:
        a[0] = -a[0]
        umul(a, b, ans)
        ans[0] = -ans[0]
        a[0] = -a[0]

    elif not sgn_a and not sgn_b:
        a[0] = -a[0]
        b[0] = -b[0]
        uadd(a, b, ans)
        a[0] = -a[0]
        b[0] = -b[0]


def add(a, b, ans):
    sgn_a = np.int32(a[0]) >= 0
    sgn_b = np.int32(b[0]) >= 0

    if sgn_a and sgn_b:
        uadd(a, b, ans)
    elif sgn_a and not sgn_b:
        b[0] = -b[0]
        if compare(a, b) >= 0:
            usub(a, b, ans)
        else:
            usub(b, a, ans)
            ans[0] = -ans[0]

        b[0] = -b[0]

    elif not sgn_a and sgn_b:
        a[0] = -a[0]
        if compare(a, b) >= 0:
            usub(a, b, ans)
            ans[0] = -ans[0]

        else:
            usub(b, a, ans)
        a[0] = -a[0]

    elif not sgn_a and sgn_b:
        a[0] = -a[0]
        b[0] = -b[0]
        uadd(a, b, ans)
        ans[0] = -ans[0]
        a[0] = -a[0]
        b[0] = -b[0]


def rsh(a, b, ans):
    n = len(a)
    ones = 2**32 - 1
    k = b % 32
    for i in reversed(range(n)):
        l = i - (b + 31) // 32
        r = i - b // 32
        al = a[l] if l >= 0 else 0
        ar = a[r] if r >= 0 else 0
        sl = (al << (32 - k)) & ones
        sr = (ar >> k) & ones
        ans[i] = sl | sr


def main():
    pi = np.array([3, 608135816, 2242054355], dtype=np.uint32)
    minus_pi = np.array([-3, 608135816, 2242054355], dtype=np.uint32)
    print(pi)
    ans = np.zeros_like(pi, dtype=np.uint32)
    rsh(pi, 64, ans)
    print(ans)
    print(decode(ans))


if __name__ == '__main__':
    main()
