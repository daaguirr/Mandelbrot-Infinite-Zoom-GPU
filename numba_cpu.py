import imageio
import numpy as np
from numba import jit

import cpu as cpu

T = 256
x0 = -0.7600189058857209
y0 = -0.0799516080512771
N = 10

ONE = cpu.encode(1, N)
FOUR = cpu.encode(4, N)
X0 = cpu.encode(x0, N)
Y0 = cpu.encode(y0, N)


def format_x(x):
    ans = str(x)
    for i in range(5 - len(ans)):
        ans = "0" + ans
    return ans


@jit(nopython=True)
def init_cpu(ans, indexes, t, n):
    tmp = np.zeros(n + 1, dtype=np.uint32)
    tmp1 = np.zeros(n + 1, dtype=np.uint32)

    for i in range(t):
        tmp.fill(0)
        cpu.umuli(indexes[i], 2, tmp)
        cpu.rsh(tmp, int(np.log2(T) + 0.5), tmp1)
        tmp.fill(0)
        cpu.sub(tmp1, ONE, tmp)
        tmp1.fill(0)

        for k in range(n + 1):
            ans[i][k] = tmp[k]


@jit(nopython=True)
def mandelbrot_api_cpu(ans, base, s, max_iters, t, n):
    for i in range(t):
        for j in range(t):

            tmp = np.zeros(n + 1, dtype=np.uint32)
            tmp1 = np.zeros(n + 1, dtype=np.uint32)
            tmp2 = np.zeros(n + 1, dtype=np.uint32)
            tmp3 = np.zeros(n + 1, dtype=np.uint32)

            cx = np.zeros(n + 1, dtype=np.uint32)
            cy = np.zeros(n + 1, dtype=np.uint32)

            zx = np.zeros(n + 1, dtype=np.uint32)
            zy = np.zeros(n + 1, dtype=np.uint32)

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
                if cpu.compare(tmp2, FOUR) > 0 and iters > 0:
                    iters -= 1
                    break
                tmp2.fill(0)
                cpu.sub(tmp, tmp1, tmp2)  # zx * zx - zy * zy

                tmp.fill(0)
                tmp1.fill(0)
                cpu.add(tmp2, cx, tmp3)  # zx * zx - zy * zy + cx;
                tmp2.fill(0)

                cpu.umuli(zx, 2, tmp)  # 2 * zx
                cpu.mul(tmp, zy, tmp1)  # 2 * zx * zy
                tmp.fill(0)
                cpu.add(tmp1, cy, zy)  # 2 * zx * zy + cy

                zx = tmp3.copy()

                tmp1.fill(0)
                tmp3.fill(0)

                iters += 1

            tmp.fill(0)
            tmp1.fill(0)
            tmp2.fill(0)

            cpu.mul(zx, zx, tmp)  # zx * zx
            cpu.mul(zy, zy, tmp1)  # zy * zy

            cpu.add(tmp, tmp1, tmp2)
            if cpu.compare(tmp2, FOUR) > 0:
                ts = iters * 1.0 / max_iters
                r = int(9 * (1 - ts) * ts * ts * ts * 255)
                g = int(15 * (1 - ts) * (1 - ts) * ts * ts * 255)
                b = int(8.5 * (1 - ts) * (1 - ts) * (1 - ts) * ts * 255)

                ans[j][i][0] = np.uint8(r)
                ans[j][i][1] = np.uint8(g)
                ans[j][i][2] = np.uint8(b)


def mandelbrot_cpu(max_iters, ss, n=N, t=T, xt=x0, yt=y0, generate=False):
    global ONE, FOUR, X0, Y0

    ONE = cpu.encode(1, N)
    FORTH = cpu.encode(4, N)

    X0 = cpu.encode(xt, n)
    Y0 = cpu.encode(yt, n)

    indexes = range(t)
    indexes = [cpu.encode(i, n) for i in indexes]
    indexes = np.array(indexes, dtype=np.uint32)

    base = np.zeros((t, n + 1), dtype=np.uint32)

    init_cpu(base, indexes, t, n)

    ans = np.zeros((t, t, 3), dtype=np.uint8)

    batch_size = 100
    batch = []
    for i, s in enumerate(ss):
        _s = cpu.encode(s, n)
        mandelbrot_api_cpu(ans, base, _s, max_iters + (i - 1) * 50, t, n)
        batch += [(i, ans.copy())]
        if generate:
            print("Progress = %f" % (i * 100 / len(ss)))
            if len(batch) == batch_size:
                for ind in range(len(batch)):
                    imn = batch[ind][0]
                    im = batch[ind][1]
                    imageio.imwrite("results/mandelbrot_cpu_%d_%d_%s.png" % (t, n, format_x(imn)), im)
                batch = []
        ans.fill(0)
    if generate:
        for ind in range(len(batch)):
            imn = batch[ind][0]
            im = batch[ind][1]
            imageio.imwrite("results/mandelbrot_cpu_%d_%d_%s.png" % (t, n, format_x(imn)), im)


if __name__ == '__main__':
    # experiment()
    # _ss = np.geomspace(0.000001, 1, 30)[::-1]
    _ss = np.array([1, 0.5], dtype=np.double)
    mandelbrot_cpu(100, _ss, t=T, generate=False)
