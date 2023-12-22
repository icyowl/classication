from contextlib import contextmanager
import time
from datetime import datetime
from numba import njit
import numpy as np
import pandas as pd
from scipy import stats
from cl.models import imjug


@contextmanager
def timer():
    t = time.time()
    yield
    print('Elapsed:', time.time() - t)



P, OUT, SAF = imjug(bet=2.25, cherry_deno=43)

def simulate(setting: int, random_state: int = 42)->tuple[float]:
    '''
    入力された設定値と、乱数シード値から遊技機のシュミレーション値を返す
    条件1: 2022年2月実績の平均アウトまで回す
    条件2: ホール割のパラメータ  bet=2.25  cherry_deno=43
    '''
    size = 9000
    out_mean = np.array([7470, 7246, 10350, 11657, 16947, 16659])  # 2022/1実績
    # out_d = dict(zip([1, 2, 3, 4, 5, 6], out.tolist()))

    xk = np.arange(len(P.T))
    pk = P[setting-1]
    im = stats.rv_discrete(name='im', values=(xk, pk))  # no numba
    sample = im.rvs(size=size, random_state=random_state)

    cum = np.cumsum([OUT[x] for x in sample])
    games = (np.abs(cum - out_mean[setting-1])).argmin()
    result = sample[:games]
    out = cum[games]
    saf = sum([SAF[x] for x in result])
    bb = (result < 3).sum()
    rb = ((result > 2) & (result < 5)).sum()

    return tuple(map(float, [bb, rb, games, out, saf]))


def hall_settings():
    dist = np.array([0.417, 0.243, 0.251, 0.046, 0.032, 0.011])  # 2022/1

    pk_odd = dist * np.array([1, 0, 1, 0, 1, 0])  # 奇数設定
    pk_even = dist * np.array([0, 1, 0, 1, 0, 1])  # 偶数設定

    pk_odd = pk_odd + np.array([-0.003, 0, 0, 0, 0, +0.01])  # 調整
    pk_even = pk_even + np.array([0.003, 0, 0, 0, 0, -0.01])

    pk_odd = pk_odd / pk_odd.sum()  # 正規化
    pk_even = pk_even / pk_even.sum()

    return pk_odd, pk_even


def even_setting_hall() -> list[float]:
    '''
    偶数設定ホールの設定配分
    return: [0.01153846 0.84615385 0.         0.12692308 0.         0.01538462]
    '''
    pk = np.array([0.452, 0.220, 0.233, 0.033, 0.048, 0.014])  # 2022/02 設定配分
    pk_even = pk * np.array([0, 1, 0, 1, 0, 1])  # 偶数設定
    pk_even = pk_even + np.array([0.003, 0, 0, 0, 0, -0.01])  # 調整
    pk_even = pk_even / pk_even.sum()  # 正規化

    return pk_even


def distribute_settings_even(pk: np.ndarray, num: int, days=30, seed=42):
    '''
        偶数設定ホールの設定配分シュミレート
        島の１か月分の設定を返す
    '''
    total_num = num * days
    dist = list(map(round, total_num * pk))
    idx = dist.index(max(dist))
    diff = total_num - sum(dist)
    dist[idx] = dist[idx] + diff  # 余りの台数を一番台数の多い設定で調整する
    # 設定1
    s1_list = [1] * dist[0] + [0] * (days - dist[0])
    # 設定2
    quotient, remainder = divmod(dist[1], days)
    s2_list = [quotient] * days
    for i in range(remainder):  # 余りの分を分配する
        s2_list[i] += 1
    # 設定6
    s6_list = [0] * days
    quotient, remainder = divmod(dist[5], 3)
    for i in [0, 10, 20]:
        s6_list[i] = quotient
    for i in range(remainder):  # 余りの分を分配する
        s6_list[i*10] += 1

    np.random.seed(seed)
    np.random.shuffle(s1_list)
    np.random.shuffle(s2_list)

    for i in [0, 10, 20]:  # 特日の設定2を設定6の分だけマイナス
        s2_list[i] -= s6_list[i]
        for j in range(s6_list[i]):  # マイナス分を分配
            s2_list[i+j+1] += 1

    arr = np.empty((days, num), dtype=np.int64)
    for i, (s1, s2, s6) in enumerate(zip(s1_list, s2_list, s6_list)):
        if num - (s1 + s2 + s6) >= 0:
            x = [1] * s1 + [2] * s2 + [6] * s6 + [4] * (num - (s1 + s2 + s6))
            np.random.seed(seed+i)
            np.random.shuffle(x)
            a = np.array(x)
            arr[i] = a
        else:
            print('foo')
    
    return arr

def one_month(arr: np.ndarray, seed: int = 0) -> pd.DataFrame:
    '''30日 シュミレーション
    arr: １か月の設定表
    '''
    rows = []
    for i, a in enumerate(arr):
        dt = datetime(2022, 1, i+1)
        for j, setting in enumerate(a):
            bb, rb, game, out, saf = simulate(setting, random_state=seed)
            row = pd.Series([j+1, bb, rb, game, out, saf], index=('no', 'bb', 'rb', 'game', 'out', 'saf'), name=dt)
            rows.append(row)
            seed += 1

    return pd.DataFrame(rows)

def core(num: int, seed: int = 0) -> None:
    pk_odd, pk_even = hall_settings()
    arr = distribute_settings_even(pk_even, num, seed=seed)
    df = one_month(arr, seed=seed)
    return df

# def twenty_four_months(num: int):
#     '''
#     '''
#     rates = []
#     for i in range(24):
#         df = one_month(num, seed=i*1000)
#         rate = df['saf'].sum() / df['out'].sum()
#         print(rate)
#         rates.append(rate)
#     return rates


if __name__ == '__main__':

    with timer():
        rates = []
        for i in range(12):
            df = core(32, seed=i*2000)
            rate = df['saf'].sum() / df['out'].sum()
            print(rate)
            rates.append(rate)
        print(np.mean(rates))
        # dist_odd, dist_even = hall_settings()
        # out = np.array([7470, 7246, 10350, 11657, 16947, 16659])  # 投入枚数
        # rate = np.array([0.975, 0.985, 1.000, 1.016, 1.038, 1.060])  # ホール機械割
        # saf = out * rate
        # print((dist_odd * saf).sum() / (dist_odd * out).sum())
        # print((dist_even * saf).sum() / (dist_even * out).sum())

