import matplotlib.pyplot as plt
import numpy as np
import itertools

T = 256

x0 = -0.7600189058857209
y0 = -0.0799516080512771
# x0 = 0
# y0 = 0
max_iters = 100


def mandelbrot(ix, iy, s):
    x = (x0 - s) + 2 * s * ix / T
    y = (y0 - s) + 2 * s * iy / T

    c = complex(x, y)
    z = complex(0, 0)
    iters = 0

    while abs(z) < 4 and iters < max_iters:
        z = z * z + c
        iters += 1
    return z, iters


def main():
    matrix = np.zeros((T, T))
    for s in np.geomspace(0.000001, 1, 30)[::-1]:
        for i, j in itertools.product(range(T), range(T)):
            _, its = mandelbrot(j, i, s=s)
            matrix[i][j] = 1 if its < max_iters else 0

        plt.matshow(matrix)
        plt.show()


if __name__ == '__main__':
    main()
