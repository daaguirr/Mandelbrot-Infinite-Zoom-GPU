import time

import imageio
import numpy as np
from numba import jit

T = 1024
from utils import x0, y0
N = 10


def measure_time(proc, **kwargs):
    start = time.time()
    result = proc(**kwargs)
    end = time.time()
    return result, end - start


@jit(nopython=True)
def mandelbrot_naive_aux(ans, base, s, max_iters, t):
    for i in range(t):
        for j in range(t):
            cx = base[i] * s + x0
            cy = base[j] * s + y0
            zx = 0
            zy = 0

            iters = 0
            while iters < max_iters:
                nzx = zx * zx - zy * zy + cx
                nzy = 2 * zx * zy + cy
                zx = nzx
                zy = nzy
                if zx * zx + zy * zy > 4.0:
                    break

                iters += 1

            # if iters < max_iters:
            #    ans[j][i][0] = np.uint8(255)
            #    ans[j][i][1] = np.uint8(255)
            #    ans[j][i][2] = np.uint8(255)

            if zx * zx + zy * zy > 4.0:
                ts = iters * 1.0 / max_iters
                r = int(9 * (1 - ts) * ts * ts * ts * 255)
                g = int(15 * (1 - ts) * (1 - ts) * ts * ts * 255)
                b = int(8.5 * (1 - ts) * (1 - ts) * (1 - ts) * ts * 255)

                ans[j][i][0] = np.uint8(r)
                ans[j][i][1] = np.uint8(g)
                ans[j][i][2] = np.uint8(b)


def format_x(x):
    ans = str(x)
    for i in range(5 - len(ans)):
        ans = "0" + ans
    return ans


def mandelbrot_naive(max_iters, ss, t=T, generate=False):
    base = np.zeros(t, dtype=np.float32)
    for i in range(t):
        base[i] = 2 * i / t - 1

    ans = np.zeros((t, t, 3), dtype=np.uint8)
    batch_size = 100
    batch = []
    for i, s in enumerate(ss):
        mandelbrot_naive_aux(ans, base, s, max_iters + (i - 1) * 50, t)
        batch += [(i, ans.copy())]
        if generate:
            print("Progress = %f" % (i * 100 / len(ss)))
            if len(batch) == batch_size:
                for ind in range(len(batch)):
                    imn = batch[ind][0]
                    im = batch[ind][1]
                    imageio.imwrite("results/mandelbrot_cpu_naive_%d_%s.png" % (t, format_x(imn)), im)
                batch = []
        ans.fill(0)
    for ind in range(len(batch)):
        imn = batch[ind][0]
        im = batch[ind][1]
        imageio.imwrite("results/mandelbrot_cpu_naive_%d_%s.png" % (t, format_x(imn)), im)


def main():
    max_iters = 400
    ss = np.geomspace(10e-30, 1, 30)[::-1]
    mandelbrot_naive(max_iters, ss, T)


if __name__ == '__main__':
    # experiment()
    _ss = np.geomspace(10e-30, 1, 240)[::-1]
    mandelbrot_naive(200, _ss, T, generate=True)
