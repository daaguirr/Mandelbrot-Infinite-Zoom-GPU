import math

import imageio
import numpy as np
from numba import cuda
import numba

import cpu
import gpu as gpu
from utils import fill_zeros, copy, N, NN

T = 256
x0 = -0.7600189058857209
y0 = -0.0799516080512771
BLOCK_SIZE = 32
log2T = int(np.log2(T) + 0.5)


def format_x(x):
    ans = str(x)
    for i in range(5 - len(ans)):
        ans = "0" + ans
    return ans


@cuda.jit
def init_gpu(ans, indexes, ONE, t, n):
    tmp = cuda.local.array(NN, numba.uint32)
    tmp1 = cuda.local.array(NN, numba.uint32)
    i = cuda.grid(1)

    fill_zeros(tmp)
    gpu.umuli(indexes[i], 2, tmp)
    gpu.rsh(tmp, log2T, tmp1)
    fill_zeros(tmp)
    gpu.sub(tmp1, ONE, tmp)
    fill_zeros(tmp1)

    for k in range(NN):
        ans[i][k] = tmp[k]


@cuda.jit
def mandelbrot_api_gpu(ans, base, s, max_iters, t, n, X0, Y0, FOUR):
    i, j = cuda.grid(2)

    ans[j][i][0] = np.uint8(0)
    ans[j][i][1] = np.uint8(0)
    ans[j][i][2] = np.uint8(0)

    tmp = cuda.local.array(NN, numba.uint32)
    tmp1 = cuda.local.array(NN, numba.uint32)
    tmp2 = cuda.local.array(NN, numba.uint32)
    tmp3 = cuda.local.array(NN, numba.uint32)

    fill_zeros(tmp)
    fill_zeros(tmp1)
    fill_zeros(tmp2)
    fill_zeros(tmp3)

    cx = cuda.local.array(NN, numba.uint32)
    cy = cuda.local.array(NN, numba.uint32)

    zx = cuda.local.array(NN, numba.uint32)
    zy = cuda.local.array(NN, numba.uint32)

    fill_zeros(cx)
    fill_zeros(cy)
    fill_zeros(zx)
    fill_zeros(zy)

    gpu.mul(base[i], s, tmp)  # s * (2 ix / T - 1)
    gpu.mul(base[j], s, tmp1)  # s * (2 iy / T - 1)

    gpu.add(tmp, X0, cx)  # s * (2 iy / T - 1) + x0
    gpu.add(tmp1, Y0, cy)  # s * (2 ix / T - 1) + x0

    fill_zeros(tmp)
    fill_zeros(tmp1)

    iters = 0

    while iters < max_iters:
        gpu.mul(zx, zx, tmp)  # zx * zx
        gpu.mul(zy, zy, tmp1)  # zy * zy

        gpu.add(tmp, tmp1, tmp2)
        if gpu.compare(tmp2, FOUR) > 0 and iters > 0:
            iters -= 1
            break
        fill_zeros(tmp2)
        gpu.sub(tmp, tmp1, tmp2)  # zx * zx - zy * zy

        fill_zeros(tmp)
        fill_zeros(tmp1)
        gpu.add(tmp2, cx, tmp3)  # zx * zx - zy * zy + cx;
        fill_zeros(tmp2)

        gpu.umuli(zx, 2, tmp)  # 2 * zx
        gpu.mul(tmp, zy, tmp1)  # 2 * zx * zy
        fill_zeros(tmp)
        gpu.add(tmp1, cy, zy)  # 2 * zx * zy + cy

        # zx = tmp3.copy()

        copy(tmp3, zx)

        fill_zeros(tmp1)
        fill_zeros(tmp3)

        iters += 1

    fill_zeros(tmp)
    fill_zeros(tmp1)
    fill_zeros(tmp2)

    gpu.mul(zx, zx, tmp)  # zx * zx
    gpu.mul(zy, zy, tmp1)  # zy * zy

    gpu.add(tmp, tmp1, tmp2)
    if gpu.compare(tmp2, FOUR) > 0:
        ts = iters * 1.0 / max_iters
        r = int(9 * (1 - ts) * ts * ts * ts * 255)
        g = int(15 * (1 - ts) * (1 - ts) * ts * ts * 255)
        b = int(8.5 * (1 - ts) * (1 - ts) * (1 - ts) * ts * 255)

        ans[j][i][0] = np.uint8(r)
        ans[j][i][1] = np.uint8(g)
        ans[j][i][2] = np.uint8(b)


def mandelbrot_gpu(max_iters, ss, n=N, t=T, xt=x0, yt=y0, generate=False):
    ONE = cuda.to_device(cpu.encode(1, n))
    FOUR = cuda.to_device(cpu.encode(4, n))

    X0 = cuda.to_device(cpu.encode(xt, n))
    Y0 = cuda.to_device(cpu.encode(yt, n))

    grid_n = math.ceil(t / BLOCK_SIZE)

    indexes = range(t)
    indexes = [cpu.encode(i, n) for i in indexes]
    indexes = np.array(indexes, dtype=np.uint32)

    base = np.zeros((t, n + 1), dtype=np.uint32)
    d_base = cuda.to_device(base)

    init_gpu[(t,), (1,)](base, indexes, ONE, t, n)

    ans = np.zeros((t, t, 3), dtype=np.uint8)
    d_ans = cuda.to_device(ans)

    batch_size = 100
    batch = []
    for i, s in enumerate(ss):
        _s = cpu.encode(s, n)
        d_s = cuda.to_device(_s)
        mandelbrot_api_gpu[(BLOCK_SIZE, BLOCK_SIZE), (grid_n, grid_n)](d_ans, d_base, d_s, max_iters + (i - 1) * 50, t,
                                                                       n, X0, Y0, FOUR)

        if generate:
            d_ans.to_host()
            batch += [(i, ans.copy())]
            import pdb
            pdb.set_trace()
            print("Progress = %f" % (i * 100 / len(ss)))
            if len(batch) == batch_size:
                for ind in range(len(batch)):
                    imn = batch[ind][0]
                    im = batch[ind][1]
                    imageio.imwrite("results/mandelbrot_gpu_%d_%d_%s.png" % (t, n, format_x(imn)), im)
                batch = []
    if generate:
        for ind in range(len(batch)):
            imn = batch[ind][0]
            im = batch[ind][1]
            imageio.imwrite("results/mandelbrot_gpu_%d_%d_%s.png" % (t, n, format_x(imn)), im)


if __name__ == '__main__':
    # experiment()
    # _ss = np.geomspace(0.000001, 1, 30)[::-1]
    _ss = np.array([1, 0.5], dtype=np.double)
    mandelbrot_gpu(100, _ss, t=T, generate=True)
