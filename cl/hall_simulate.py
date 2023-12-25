import numpy as np
from numba import njit, jit
import pandas as pd
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
def simulate(setting: int, seed=0) -> np.ndarray:
    '''
    入力された設定値と、乱数シード値から遊技機のシュミレーション値を返す
    条件1: 2022年2月実績のアウト(target_outs)まで回す
    条件2: 実績のアウトとその一番近い数値の差が、8枚以上の場合、...
    条件3: ホール割のパラメータ  bet=2.25  cherry_deno=43
    '''
    size = 9000
    target_out = np.array([7470, 7246, 10350, 11657, 16947, 16659])  # 2022/1実績

    xk = np.arange(P.shape[1])
    pk = P[setting-1]
    # im = stats.rv_discrete(name='im', values=(xk, pk))  # no numba
    # samples = im.rvs(size=size, random_state=seed)
    np.random.seed(seed=seed)
    uniform_samples = np.random.uniform(0, 1, size=size)
    cdf = pk.cumsum()
    cdf /= cdf[-1]
    samples = np.searchsorted(cdf, uniform_samples)

    seq = [OUT[x] for x in samples]
    cum = np.array(seq).cumsum()
    t_out = target_out[setting - 1]
    idx = (np.abs(cum - t_out)).argmin()

    out = cum[idx]
    # 確率的な説明は?
    if abs(out - t_out) > 3 and seq[idx] < 21.25:  # ボーナスは取り切る
        idx += 1
        out = cum[idx]

    result = samples[:idx+1]
    saf = sum([SAF[x] for x in result])
    game = idx
    bb = (result < 3).sum()
    rb = ((result > 2) & (result < 5)).sum()

    return np.array([bb, rb, game, out, saf], dtype=np.float64)

@njit(cache=True)
def loop(setting: int, size: int, step=10):
    rows = []
    for i in range(size):
        arr = simulate(setting, seed=i*step)
        rows.append(arr)
    return rows

if __name__ == '__main__':
    
    with timer():
        for i in range(1, 7):
            rows = loop(i, 100000, step=10)
            df = pd.DataFrame(rows, columns=('bb', 'rb', 'game', 'out', 'saf'))
            print(i, df['saf'].sum() / df['out'].sum())