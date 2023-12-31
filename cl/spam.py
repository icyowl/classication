import numpy as np
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import time

@contextmanager
def timer():
    t = time.time()
    yield
    print('Elapsed:', time.time() - t)


def imjug(bet=3., cherry_deno=39.35):
    big = 431.16, 422.81, 422.81, 417.43, 417.43, 407.06
    cbig = 1129.93, 1129.93, 1129.93, 1092.27, 1092.27, 1092.27
    rbig = 2184.53, 2184.53, 2184.53, 1820.44, 1820.44, 1820.44
    reg = 642.51, 590.41, 474.90, 448.88, 364.09, 364.09
    creg = 1394.38, 1236.53, 1092.27, 1057.03, 851.12, 851.12
    bell =  (1092.27,) * 6
    clown = (1092.27,) * 6
    grape = 6.02, 6.02, 6.02, 6.02, 6.02, 5.78
    cherry =  (32.28,) * 6
    replay = (7.298,) * 6

    p = 1/np.array([big, cbig, rbig, reg, creg, bell, clown, grape, cherry, replay])
    p *= np.array([1., 1., 1., 1., 1., 1092.27/5734, 1092.27/14167, 1., 32.28/cherry_deno, 1.]).reshape(-1, 1)
    q = 1 - np.sum(p, axis=0).reshape(1, -1)
    pk = np.concatenate([p, q]).T

    out = np.array([3+42+bet, 3+42+bet, 3+42+bet, 3+16+bet, 3+16+bet, 3., 3., 3., 3., 3., 3.])
    saf = np.array([294., 294+2., 294+1., 112., 112+2., 14., 10., 8., 2., 3., 0.])
    saf += np.array([0., 2*14/21-2, 14/21-1, 2*14/21-2, 0., 2*14/21-2, 0., 0., 0., 0., 0.])

    return pk, out, saf

pk, out, saf = imjug()

def func(i, size):
    pk_ = pk[i-1]
    np.random.seed(seed=42)
    uniform_samples = np.random.uniform(0, 1, size=size)
    cdf = pk_.cumsum()
    cdf /= cdf[-1]
    samples = np.searchsorted(cdf, uniform_samples)
    hist, _ = np.histogram(samples, bins=pk_.size)
    return i, np.sum(saf*hist) / np.sum(out*hist) * 100

def run(size):
    executor = ThreadPoolExecutor(max_workers=6)
    futures = []
    for i in range(1, 7):
        future = executor.submit(func, i, size)
        futures.append(future)

    for f in futures:
        print(*f.result())

with timer():
    run(size=8000*10000)