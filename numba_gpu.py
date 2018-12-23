import math

import imageio
import numba
import numpy as np
from numba import cuda

import cpu
import gpu as gpu
from utils import fill_zeros, copy, N, NN, log2T, T, x0, y0, BLOCK_SIZE


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
    fill_zeros(tmp1)
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
    

    if iters < max_iters:
        #ans[j][i][0] = np.uint8(255)
        #ans[j][i][1] = np.uint8(255)
        #ans[j][i][2] = np.uint8(255)
        nit = iters % 16
        if  nit == 0 :
        	ans[j][i][0] = np.uint8(66)
        	ans[j][i][1] = np.uint8(30)
        	ans[j][i][2] = np.uint8(15)
        if  nit == 1 :
        	ans[j][i][0] = np.uint8(25)
        	ans[j][i][1] = np.uint8(7)
        	ans[j][i][2] = np.uint8(26)
        if  nit == 2 :
        	ans[j][i][0] = np.uint8(9)
        	ans[j][i][1] = np.uint8(1)
        	ans[j][i][2] = np.uint8(47)
        if  nit == 3 :
        	ans[j][i][0] = np.uint8(4)
        	ans[j][i][1] = np.uint8(4)
        	ans[j][i][2] = np.uint8(73)
        if  nit == 4 :
        	ans[j][i][0] = np.uint8(0)
        	ans[j][i][1] = np.uint8(7)
        	ans[j][i][2] = np.uint8(100)
        if  nit == 5 :
        	ans[j][i][0] = np.uint8(12)
        	ans[j][i][1] = np.uint8(44)
        	ans[j][i][2] = np.uint8(138)
        if  nit == 6 :
        	ans[j][i][0] = np.uint8(24)
        	ans[j][i][1] = np.uint8(82)
        	ans[j][i][2] = np.uint8(177)
        if  nit == 7 :
        	ans[j][i][0] = np.uint8(57)
        	ans[j][i][1] = np.uint8(125)
        	ans[j][i][2] = np.uint8(209)
        if  nit == 8 :
        	ans[j][i][0] = np.uint8(134)
        	ans[j][i][1] = np.uint8(181)
        	ans[j][i][2] = np.uint8(229)
        if  nit == 9 :
        	ans[j][i][0] = np.uint8(211)
        	ans[j][i][1] = np.uint8(236)
        	ans[j][i][2] = np.uint8(248)
        if  nit == 10 :
        	ans[j][i][0] = np.uint8(241)
        	ans[j][i][1] = np.uint8(233)
        	ans[j][i][2] = np.uint8(191)
        if  nit == 11 :
        	ans[j][i][0] = np.uint8(248)
        	ans[j][i][1] = np.uint8(201)
        	ans[j][i][2] = np.uint8(95)
        if  nit == 12 :
        	ans[j][i][0] = np.uint8(255)
        	ans[j][i][1] = np.uint8(170)
        	ans[j][i][2] = np.uint8(0)
        if  nit == 13 :
        	ans[j][i][0] = np.uint8(204)
        	ans[j][i][1] = np.uint8(128)
        	ans[j][i][2] = np.uint8(0)
        if  nit == 14 :
        	ans[j][i][0] = np.uint8(153)
        	ans[j][i][1] = np.uint8(87)
        	ans[j][i][2] = np.uint8(0)
        if  nit == 15 :
        	ans[j][i][0] = np.uint8(106)
        	ans[j][i][1] = np.uint8(52)
        	ans[j][i][2] = np.uint8(3)
    #gpu.mul(zx, zx, tmp)  # zx * zx
    
    #gpu.mul(zy, zy, tmp1)  # zy * zy

    #gpu.add(tmp, tmp1, tmp2)
    #if gpu.compare(tmp2, FOUR) > 0:
    #    ts = iters * 1.0 / max_iters
    #    r = int(9 * (1 - ts) * ts * ts * ts * 255)
    #    g = int(15 * (1 - ts) * (1 - ts) * ts * ts * 255)
    #    b = int(8.5 * (1 - ts) * (1 - ts) * (1 - ts) * ts * 255)
    
    #
    #    ans[j][i][0] = np.uint8(r)
    #    ans[j][i][1] = np.uint8(g)
    #    ans[j][i][2] = np.uint8(b)


def mandelbrot_gpu(max_iters, ss, n=N, t=T, xt=x0, yt=y0, generate=False):
    ONE = cuda.to_device(cpu.encode(1, n))
    FOUR = cuda.to_device(cpu.encode(4, n))

    #X0 = cuda.to_device(cpu.encode(xt, n))
    #Y0 = cuda.to_device(cpu.encode(yt, n))

    #X0 = cuda.to_device(np.array([0, 889521515,3920901292,2551795869,690843754,1095124392,3810045335,2280915385,3933404175,116643750, 1], dtype=np.uint32))
    #Y0 = cuda.to_device(np.array([1,527237573,2813191363,3485787269,718281163,2930215389,2988428924,1688274651,3514889844,4244049646, 1], dtype=np.uint32))
    X0 = cuda.to_device(np.array([         1, 3178543730,  764955228, 1653432584, 1289000324,
       1853942647, 1801344735, 4044992822, 2925897450, 3614139489,
                1], dtype=np.uint32))
    Y0 = cuda.to_device(np.array([         0,  121012162, 3888660452, 2098384151, 2824235263,
        553223273, 1987301639, 1103228041,  141260832, 3674123863,
                1], dtype=np.uint32))
    grid_n = math.ceil(t / BLOCK_SIZE)

    base = np.zeros(t, dtype=np.double)
    for i in range(t):
        base[i] = 2 * i / t - 1
    base = [cpu.encode(b, n) for b in base]
    # indexes = range(t)
    # indexes = [cpu.encode(i, n) for i in indexes]
    # indexes = np.array(indexes, dtype=np.uint32)
    # d_indexes = cuda.to_device(indexes)

    # base = np.zeros((t, n + 1), dtype=np.uint32)
    d_base = cuda.to_device(base)

    # init_gpu[(t,), (1,)](d_base, d_indexes, ONE, t, n)

    #d_base.to_host()
    # print(base)

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
    _ss = np.geomspace(10e-60, 1, 900, dtype=np.double)[::-1]
    #_ss = np.array([1, 0.5], dtype=np.double)
    mandelbrot_gpu(2000, _ss, t=T, generate=True)
