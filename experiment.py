import math
import time

import imageio
import numpy as np

import cpu
from numba_cpu import mandelbrot_cpu
from numba import cuda
import numba

BITS = 32
w = np.uint64(1 << 32)
ones = np.uint32(w - 1)
x0 = -0.7600189058857209
y0 = -0.0799516080512771
BLOCK_SIZE = 32

N = 3
NN = 4
T = 256
log2T = int(np.log2(T) + 0.5)


@cuda.jit(device=True)
def uadd(a, b, ans):
    carry = np.uint64(0)
    for i in range(N, -1, -1):
        abc = np.uint64(a[i]) + np.uint64(b[i]) + carry
        ans[i] = abc
        carry = np.uint64(abc >> BITS)


@cuda.jit(device=True)
def usub(a, b, ans):
    aux = cuda.local.array(NN, numba.uint32)
    copy(b, aux)
    i = N - 1
    while i >= 0 and b[i] == 0:
        aux[i] = ones
        i -= 1
    if i >= 0:
        aux[i] -= 1
    for j in range(N):
        aux[j] ^= ones
    uadd(a, aux, ans)


@cuda.jit(device=True)
def ucompare(a, b):
    for i in range(N):
        if a[i] != b[i]:
            return np.int64(a[i]) - np.int64(b[i])
    return np.int64(0)


@cuda.jit(device=True)
def compare(a, b):
    if a[N] != b[N]:
        return np.int64(b[N]) - np.int64(a[N])
    return ucompare(a, b)


@cuda.jit(device=True)
def lsh32(a, b):
    for i in range(N):
        if i <= N - 1 - b:
            a[i] = a[i + b]
        else:
            a[i] = 0


@cuda.jit(device=True)
def rsh32(a, b):
    for i in range(N, -1, -1):
        if i - b >= 0:
            a[i] = a[i - b]
        else:
            a[i] = 0


@cuda.jit(device=True)
def umuli(a, b, ans):
    b = np.uint64(b)
    carry = np.uint64(0)
    for i in range(N, -1, -1):
        abc = np.uint64(a[i]) * b + carry
        ans[i] = abc
        carry = np.uint64(abc >> BITS)
    ans[N] = a[N]


@cuda.jit(device=True)
def mul(a, b, ans):
    aux = cuda.local.array(NN, numba.uint32)
    tmp = cuda.local.array(NN, numba.uint32)
    tmp2 = cuda.local.array(NN, numba.uint32)

    copy(a, aux)
    fill_zeros(tmp)
    fill_zeros(tmp2)

    for i in range(N):
        umuli(aux, b[i], tmp)
        uadd(ans, tmp, tmp2)
        copy(tmp2, ans)
        rsh32(aux, 1)
    ans[N] = a[N] ^ b[N]


@cuda.jit(device=True)
def add(a, b, ans):
    if a[N] == b[N]:
        uadd(a, b, ans)
        ans[N] = a[N]
    elif ucompare(a, b) >= 0:
        usub(a, b, ans)
        ans[N] = a[N]
    else:
        usub(b, a, ans)
        ans[N] = b[N]


# arithmetic shift
@cuda.jit(device=True)
def rsh(a, b, ans):
    k = b % BITS
    for i in range(N, -1, -1):
        l = i - (b + BITS - 1) // BITS
        r = i - b // BITS
        al = a[l] if l >= 0 else 0
        ar = a[r] if r >= 0 else 0
        sl = (al << (32 - k)) & ones
        sr = (ar >> k) & ones
        ans[i] = sl | sr
    ans[N] = a[N]


@cuda.jit(device=True)
def sub(a, b, ans):
    if a[N] != b[N]:
        uadd(a, b, ans)
        ans[N] = a[N]
    elif ucompare(a, b) >= 0:
        usub(a, b, ans)
        ans[N] = a[N]
    else:
        usub(b, a, ans)
        ans[N] = a[N] ^ 1


@cuda.jit(device=True)
def fill_zeros(a):
    for i in range(N + 1):
        a[i] = 0


@cuda.jit(device=True)
def copy(source, target):
    for i in range(N + 1):
        target[i] = source[i]


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
    umuli(indexes[i], 2, tmp)
    rsh(tmp, log2T, tmp1)
    fill_zeros(tmp)
    sub(tmp1, ONE, tmp)
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

    mul(base[i], s, tmp)  # s * (2 ix / T - 1)
    mul(base[j], s, tmp1)  # s * (2 iy / T - 1)

    add(tmp, X0, cx)  # s * (2 iy / T - 1) + x0
    add(tmp1, Y0, cy)  # s * (2 ix / T - 1) + x0

    fill_zeros(tmp)
    fill_zeros(tmp1)

    iters = 0

    while iters < max_iters:
        mul(zx, zx, tmp)  # zx * zx
        mul(zy, zy, tmp1)  # zy * zy

        add(tmp, tmp1, tmp2)
        if compare(tmp2, FOUR) > 0 and iters > 0:
            iters -= 1
            break
        fill_zeros(tmp2)
        sub(tmp, tmp1, tmp2)  # zx * zx - zy * zy

        fill_zeros(tmp)
        fill_zeros(tmp1)
        add(tmp2, cx, tmp3)  # zx * zx - zy * zy + cx;
        fill_zeros(tmp2)

        umuli(zx, 2, tmp)  # 2 * zx
        mul(tmp, zy, tmp1)  # 2 * zx * zy
        fill_zeros(tmp)
        add(tmp1, cy, zy)  # 2 * zx * zy + cy

        # zx = tmp3.copy()

        copy(tmp3, zx)

        fill_zeros(tmp1)
        fill_zeros(tmp3)

        iters += 1

    fill_zeros(tmp)
    fill_zeros(tmp1)
    fill_zeros(tmp2)

    mul(zx, zx, tmp)  # zx * zx
    mul(zy, zy, tmp1)  # zy * zy

    add(tmp, tmp1, tmp2)
    if compare(tmp2, FOUR) > 0:
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
    d_indexes = cuda.to_device(indexes)

    base = np.zeros((t, n + 1), dtype=np.uint32)
    d_base = cuda.to_device(base)

    init_gpu[(t,), (1,)](d_base, d_indexes, ONE, t, n)

    #d_base.to_host()
    #print(base)

    ans = np.zeros((t, t, 3), dtype=np.uint8)
    d_ans = cuda.to_device(ans)

    batch_size = 100
    batch = []
    for i, s in enumerate(ss):
        _s = cpu.encode(s, n)
        d_s = cuda.to_device(_s)
        # mandelbrot_api_gpu[(BLOCK_SIZE, BLOCK_SIZE), (grid_n, grid_n)](d_ans, d_base, d_s, max_iters + (i - 1) * 50, t,
        #                                                               n, X0, Y0, FOUR)
        mandelbrot_api_gpu[(BLOCK_SIZE, BLOCK_SIZE), (grid_n, grid_n)](d_ans, d_base, d_s, max_iters, t,
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


def measure_time(proc, **kwargs):
    start = time.time()
    result = proc(**kwargs)
    end = time.time()
    return result, end - start


def experiment():
    global N, NN, T, log2T
    ts = [16, 32, 64, 128, 256, 512, 2015]
    max_iters = 200
    ss = np.geomspace(0.000001, 1, 30)[::-1]
    ns = range(5, 10)
    count = 0
    with open('results.csv', 'wb') as file:
        file.write("t,n,device,fps\n".encode('utf8'))
        for i, n in enumerate(ns):
            for j, t in enumerate(ts):
                N = n
                T = t
                NN = N + 1
                log2T = int(np.log2(T) + 0.5)

                print(f"Progress = {count * 100 / (len(ns) * len(ts))}%")
                _, delta = measure_time(mandelbrot_cpu, max_iters=max_iters, ss=ss, n=n, t=t, xt=x0, yt=y0,
                                        generate=False)
                file.write(f"{t},{n},{0},{len(ss) / delta}\n".encode('utf8'))
                _, delta = measure_time(mandelbrot_gpu, max_iters=max_iters, ss=ss, t=t, n=n)
                file.write(f"{t},{n},{1},{len(ss) / delta}\n".encode('utf8'))
                file.flush()
                count += 1
