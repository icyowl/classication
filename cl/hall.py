import numpy as np
from numba import njit
from scipy import stats
from cl.models import imjug

from contextlib import contextmanager
import time
@contextmanager
def timer():
    t = time.time()
    yield
    print('Elapsed:', time.time() - t)

P, OUT, SAF = imjug(bet=2.25, cherry_deno=43)

@njit(cache=True)
def simulate(setting: int, seed=0) -> list[float]:
    '''
    入力された設定値と、乱数シード値から遊技機のシュミレーション値を返す
    条件1: 2022年2月実績のアウト(target_outs)まで回す
    条件2: ホール割のパラメータ  bet=2.25  cherry_deno=43
    '''
    size = 9000
    target_outs = np.array([7470, 7246, 10350, 11657, 16947, 16659])  # 2022/1実績
    # out_d = dict(zip([1, 2, 3, 4, 5, 6], out.tolist()))

    xk = np.arange(P.shape[1])
    pk = P[setting-1]
    # im = stats.rv_discrete(name='im', values=(xk, pk))  # no numba
    # samples = im.rvs(size=size, random_state=seed)
    np.random.seed(seed)
    uniform_samples = np.random.uniform(0, 1, size=size)
    cdf = pk.cumsum()
    cdf /= cdf[-1]
    samples = np.searchsorted(cdf, uniform_samples)

    out_ = [OUT[x] for x in samples]
    cum = np.array(out_).cumsum()
    t_out = target_outs[setting - 1]
    games = (np.abs(cum - t_out)).argmin()
    result = samples[:games]

    out = cum[games-1]
    saf = sum([SAF[x] for x in result])
    bb = (result < 3).sum()
    rb = ((result > 2) & (result < 5)).sum()

    return list(map(float, [bb, rb, games, out, saf]))


@njit(cache=True)
def core(setting: int, size: int):
    outs, safs = [], []
    for i in range(size):
        bb, rb, games, out, saf = simulate(setting, seed=i)
        outs.append(out)
        safs.append(saf)
    print(sum(safs) / sum(outs))

if __name__ == '__main__':
    
    with timer():
        core(1, 100000)