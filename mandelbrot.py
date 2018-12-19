import matplotlib.pyplot as plt
import numpy as np
import itertools
import cpu as cpu

T = 256

x0 = -0.7600189058857209
y0 = -0.0799516080512771

N = 10
X0 = cpu.encode(x0, N)
Y0 = cpu.encode(y0, N)
NEG_ONE = cpu.encode(-1, N)
TWO = cpu.encode(2, N)


def mandelbrot_naive(ix, iy, s, max_iters=100):
    x = (x0 - s) + 2 * s * ix / T
    y = (y0 - s) + 2 * s * iy / T

    c = complex(x, y)
    z = complex(0, 0)
    iters = 0

    while abs(z) < 4 and iters < max_iters:
        z = z * z + c
        iters += 1
    return z, iters


def mandelbrot_api_cpu(ix, iy, s, max_iters=100):
    _ix = cpu.encode(ix, N)
    _iy = cpu.encode(iy, N)
    _s = cpu.encode(s, N)

    tmp = np.zeros_like(_s)
    tmp1 = np.zeros_like(_s)
    tmp2 = np.zeros_like(_s)

    # -s
    cpu.mul(_s, NEG_ONE, tmp)
    _ms = tmp.copy()
    tmp.fill(0)

    # ------------------------------------------------ #
    # X0 - s
    cpu.add(X0, _ms, tmp)

    # 2 * s * ix / T
    cpu.mul(TWO, _s, tmp1)
    cpu.mul(tmp1, _ix, tmp2)
    tmp1.fill(0)
    cpu.rsh(tmp2, int(np.log2(T) + 0.5), tmp1)
    tmp2.fill(0)
    # (x0 - s) + 2 * s * ix / T
    cpu.add(tmp, tmp1, tmp2)
    x = tmp2.copy()

    # clean
    tmp.fill(0)
    tmp1.fill(0)
    tmp2.fill(0)

    # ------------------------------------------------ #
    # Y0 - s
    cpu.add(Y0, _ms, tmp)

    # 2 * s * ix / T
    cpu.mul(TWO, _s, tmp1)
    cpu.mul(tmp1, _iy, tmp2)
    tmp1.fill(0)
    cpu.rsh(tmp2, int(np.log2(T) + 0.5), tmp1)
    tmp2.fill(0)
    # (x0 - s) + 2 * s * ix / T
    cpu.add(tmp, tmp1, tmp2)
    y = tmp2.copy()


def main():
    matrix = np.zeros((T, T))
    max_iters = 100
    for s in np.geomspace(0.000001, 1, 30)[::-1]:
        for i, j in itertools.product(range(T), range(T)):
            _, its = mandelbrot_naive(j, i, s=s, max_iters=max_iters)
            matrix[i][j] = 1 if its < max_iters else 0

        plt.matshow(matrix)
        plt.show()


if __name__ == '__main__':
    main()
