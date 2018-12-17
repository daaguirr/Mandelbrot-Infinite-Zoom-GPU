from uadd import uadd


def usub(a, b, ans):
    n = len(a)
    i = n-1
    while i >= 0 and b[i] == 0:
        b[i] = -1
        i -= 1
    b[i] ^= 2**32 - 1
    uadd(a, b, ans)
