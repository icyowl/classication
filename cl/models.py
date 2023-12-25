import numpy as np
from numba import njit
from scipy import stats
np.set_printoptions(suppress=True, precision=7)

from contextlib import contextmanager
import time
@contextmanager
def timer():
    t = time.time()
    yield
    print('Elapsed:', time.time() - t)

@njit(cache=True)
def imjug(bet: float = 2.25, cherry_deno: float = 48.42):
    # 引数のデフォルト： ボーナス揃え 3bet 先ペカ考慮なし、チェリー 14/21 x 1/32.28 = 1/48.42
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
    role = 1/np.array([big, cbig, rbig, reg, creg, bell, clown, grape, cherry, replay])
    # 適当押し： ベル  1/5734, ピエロ 1/14167, チェリー 1/48.42 (default) へ変換
    role = role * np.array([1., 1., 1., 1., 1., 1092.27/5734, 1092.27/14167, 1., 32.28/cherry_deno, 1.]).reshape(-1, 1)
    # loss = (1 - np.sum(role, axis=0))  # .reshape(1, -1)
    # p = np.concatenate([role, loss]).T  # no numba
    q = np.empty((11, 6), dtype=np.float64)
    q[0:10, :] = role
    q[10, :] = 1 - np.sum(role, axis=0)
    p = q.T

    out = np.array([3+42+bet, 3+42+bet, 3+42+bet, 3+16+bet, 3+16+bet, 3., 3., 3., 3., 3., 3.], dtype=np.float64)
    saf = np.array([294., 294+2., 294+1., 112., 112+2., 14., 10., 8., 2., 3., 0.], dtype=np.float64)
    # 適当押し： 重複チェリーの期待枚数は、払い出し x 14/21
    saf = saf + np.array([0., -2+(2*14/21), -1+(14/21), -2+(2*14/21), 0., -2+(2*14/21), 0., 0., 0., 0., 0.], dtype=np.float64)

    return p, out, saf


def main(size=8000, seed=42):

    p, out, saf = imjug(bet=2.25, cherry_deno=43)

    for i in range(1, 7):
        xk = np.arange(p.shape[1])
        pk = p[i-1]
        im = stats.rv_discrete(name='im', values=(xk, pk)) 
        samples = im.rvs(size=size, random_state=seed)
        outs = [out[x] for x in samples]
        safs = [saf[x] for x in samples]
        print(i, sum(safs) / sum(outs))

@njit(cache=True)
def core(size=8000, seed=42):

    p, out, saf = imjug(bet=2.25, cherry_deno=43)

    for i in range(1, 7):
        # xk = np.arange(p.shape[1])
        pk = p[i-1]
        np.random.seed(seed=seed)
        uniform_samples = np.random.uniform(0, 1, size=size)
        cdf = pk.cumsum()
        cdf /= cdf[-1]
        samples = np.searchsorted(cdf, uniform_samples)
        outs = [out[x] for x in samples]
        safs = [saf[x] for x in samples]
        print(i, sum(safs) / sum(outs))

if __name__ == '__main__':
    
    with timer():
        # main(size=8000 * 1000)
        core(size=8000 * 10000)
