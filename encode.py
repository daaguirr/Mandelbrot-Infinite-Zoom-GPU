import numpy as np


def encode(x, n):
    was_negative = x < 0
    x = abs(x)

    ans = np.empty(n, dtype=np.uint32)
    for i in range(n):
        ans[i] = int(x)
        x = (x - int(x)) * 2**32

    if was_negative:
        ans[0] = -ans[0]
    return ans


def main():
    print(encode(np.pi, 5))


if __name__ == '__main__':
    main()
