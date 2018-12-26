import time

import imageio
import numpy as np

import cpu as cpu

T = 256
x0 = -0.7600189058857209
y0 = -0.0799516080512771
N = 10

NEG_ONE = cpu.encode(-1, N)
TWO = cpu.encode(2, N)
FOUR = cpu.encode(4, N)


def measure_time(proc, **kwargs):
    start = time.time()
    result = proc(**kwargs)
    end = time.time()
    return result, end - start


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


def mandelbrot_api_cpu(ans, base, s, max_iters, t, n, X0, Y0):
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
                if cpu.compare(tmp2, FOUR) > 0:
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

                ans[j][i][0] = r
                ans[j][i][1] = g
                ans[j][i][2] = b


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

            if zx * zx + zy * zy > 4.0:
                ts = iters * 1.0 / max_iters
                r = int(9 * (1 - ts) * ts * ts * ts * 255)
                g = int(15 * (1 - ts) * (1 - ts) * ts * ts * 255)
                b = int(8.5 * (1 - ts) * (1 - ts) * (1 - ts) * ts * 255)

                ans[j][i][0] = np.uint8(r)
                ans[j][i][1] = np.uint8(g)
                ans[j][i][2] = np.uint8(b)


def format_x(x):
    return ('0000' + str(x))[-5:]


def mandelbrot_naive(max_iters, ss, t=T, generate=False):
    base = np.zeros(t, dtype=np.double)
    for i in range(t):
        base[i] = 2 * i / t - 1

    ans = np.zeros((t, t, 3), dtype=np.uint8)
    for i, s in enumerate(ss):
        mandelbrot_naive_aux(ans, base, s, max_iters, t)
        # print(ans)
        # plt.matshow(ans)
        # plt.show()

        if generate:
            print(f"Progress = {i * 100/len(ss)} %")
            imageio.imwrite(f"results/mandelbrot_cpu_naive_{t}_{format_x(i)}.png", ans)
        ans.fill(0)


def mandelbrot_cpu(max_iters, ss, n=N, t=T, xt=x0, yt=y0, generate=False):
    X0 = cpu.encode(xt, n)
    Y0 = cpu.encode(yt, n)

    indexes = range(t)
    indexes = [cpu.encode(i, n) for i in indexes]
    indexes = np.array(indexes, dtype=np.uint32)

    base = np.zeros((t, n), dtype=np.uint32)
    init_cpu(base, indexes, t, n)

    ans = np.array((t, t))

    for i, s in enumerate(ss):
        _s = cpu.encode(s, n)
        mandelbrot_api_cpu(ans, base, _s, max_iters, t, n, X0, Y0)
        if generate:
            print(f"Progress = {i * 100 / len(ss)} %")
            imageio.imwrite(f"results/mandelbrot_cpu_{t}_{n}_{format_x(i)}.npy", ans)
        ans.fill(0)


def mandelbrot_gpu(max_iters, ss, n=N, t=T, xt=x0, yt=y0, generate=False):
    for i, s in enumerate(ss):
        time.sleep(t * n * i * 0.0001)


def main():
    max_iters = 100
    ss = np.geomspace(0.000001, 1, 30)[::-1]
    mandelbrot_naive(max_iters, ss, T)


def experiment():
    ts = [16, 32, 64, 128]
    max_iters = 100
    ss = np.geomspace(0.000001, 1, 15)[::-1]
    ns = range(5, 10)
    count = 0
    with open('results.csv', 'wb') as file:
        file.write("t,n,device,fps\n".encode('utf8'))
        for i, n in enumerate(ns):
            for j, t in enumerate(ts):
                print(f"Progress = {count * 100 / (len(ns) * len(ts))}%")
                _, delta = measure_time(mandelbrot_naive, max_iters=max_iters, ss=ss, t=t)
                file.write(f"{t},{n},{0},{len(ss) / delta}\n".encode('utf8'))
                _, delta = measure_time(mandelbrot_gpu, max_iters=max_iters, ss=ss, t=t, n=n)
                file.write(f"{t},{n},{1},{len(ss) / delta}\n".encode('utf8'))
                file.flush()
                count += 1


if __name__ == '__main__':
    # experiment()
    _ss = np.geomspace(0.000001, 1, 30)[::-1]
    mandelbrot_naive(100, _ss, T, generate=True)
