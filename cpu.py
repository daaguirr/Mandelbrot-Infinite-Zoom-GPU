import numpy as np


def iumul(a, b):
    ans = 0
    while b:
        if b & 1:
            ans += a
        a += a
        b >>= 1
    return ans


print(iumul(10, 2))


def main():
    pass


if __name__ == '__main__':
    main()
