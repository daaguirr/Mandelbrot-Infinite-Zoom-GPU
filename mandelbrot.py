import matplotlib.pyplot as plt
import numpy as np
import itertools
import cpu as cpu

T = 256
x0 = -0.7600189058857209
y0 = -0.0799516080512771
N = 10

NEG_ONE = cpu.encode(-1, N)
TWO = cpu.encode(2, N)
FORTH = cpu.encode(4, N)


def mandelbrot_naive(ix, iy, s, max_iters=100):
    x = (x0 - s) + 2 * s * ix / T  # s * (2 ix / T - 1) + x0
    y = (y0 - s) + 2 * s * iy / T

    c = complex(x, y)
    z = complex(0, 0)
    iters = 0

    while abs(z) < 4 and iters < max_iters:
        z = z * z + c
        iters += 1
    return z, iters


def init_cpu(ans, indexes, t, n):
    tmp = np.zeros(n, dtype=np.uint32)
    tmp1 = np.zeros(n, dtype=np.uint32)

    for i in range(t):
        cpu.mul(TWO, indexes[i], tmp)
        cpu.rsh(tmp, int(np.log2(T) + 0.5), tmp1)
        tmp.fill(0)
        cpu.add(tmp1, NEG_ONE, tmp)
        tmp1.fill(0)

        for k in range(n + 1):
            ans[i][k] = tmp[k]


def mandelbrot_api_cpu(ans, base, s, max_iters, t, X0, Y0, n):
    for i in range(t):
        for j in range(t):

            tmp = np.zeros(n, dtype=np.uint32)
            tmp1 = np.zeros(n, dtype=np.uint32)
            tmp2 = np.zeros(n, dtype=np.uint32)
            tmp3 = np.zeros(n, dtype=np.uint32)

            cx = np.zeros(n, dtype=np.uint32)
            cy = np.zeros(n, dtype=np.uint32)

            zx = np.zeros(n, dtype=np.uint32)
            zy = np.zeros(n, dtype=np.uint32)

            cpu.mul(base[i], s, tmp)  # s * (2 ix / T - 1)
            cpu.mul(base[j], s, tmp1)  # s * (2 iy / T - 1)

            cpu.add(tmp, X0, cx)  # s * (2 iy / T - 1) + x0
            cpu.add(tmp1, Y0, cy)  # s * (2 ix / T - 1) + x0

            tmp.fill(0)
            tmp1.fill(0)

            iters = 0

            while iters < max_iters:
                cpu.mul(zx, zx, tmp)  # zx * zx
                cpu.mul(zy, zy, tmp1)  # zy * zy

                cpu.add(tmp, tmp1, tmp2)
                if cpu.compare(tmp2, FORTH) > 0:
                    if iters > 0:
                        iters -= 1
                    break

                tmp2.fill(0)
                cpu.mul(tmp1, NEG_ONE, tmp2)  # -(zy * zy)
                tmp1.fill(0)
                cpu.add(tmp, tmp2, tmp1)  # zx * zx - zy * zy
                tmp.fill(0)
                tmp2.fill(0)
                cpu.add(tmp1, cx, tmp3)  # zx * zx - zy * zy + cx;
                tmp1.fill(0)

                cpu.mul(TWO, zx, tmp)  # 2 * zx
                cpu.mul(tmp, zy, tmp1)  # 2 * zx * zy
                tmp.fill(0)
                cpu.add(tmp1, cy, zy)  # 2 * zx * zy + cy

                zx = tmp3.copy()

                tmp1.fill(0)
                tmp3.fill(0)

                iters += 1

            ans[i][j] = 1 if iters < max_iters else 0


def mandelbrot_cpu(max_iters, ss, n=N, t=T, xt=x0, yt=y0):
    X0 = cpu.encode(xt, n)
    Y0 = cpu.encode(yt, n)

    indexes = range(t)
    indexes = [cpu.encode(i, n) for i in indexes]
    indexes = np.array(indexes, dtype=np.uint32)

    base = np.zeros((t, n), dtype=np.uint32)
    init_cpu(base, indexes, t, n)

    ans = np.array((t, t))
    for s in ss:
        _s = cpu.encode(s, n)
        mandelbrot_api_cpu(ans, base, _s, max_iters, t, X0, Y0, n)


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
