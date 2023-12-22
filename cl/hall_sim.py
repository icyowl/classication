from contextlib import contextmanager
import time
from datetime import datetime
from numba import njit
import numpy as np
import pandas as pd
from scipy import stats


@contextmanager
def timer():
    t = time.time()
    yield
    print('Elapsed:', time.time() - t)

hallrate = 0.975, 0.985, 1.000, 1.016, 1.038, 1.060

=======
import time


@contextmanager
def timer():
    t = time.perf_counter()
    yield None
    print('Elapsed:', time.perf_counter() - t)


def imjug(bet=3, cherry_deno=48.42):
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
    role = role * np.array([1, 1, 1, 1, 1, 1092.27/5734, 1092.27/14167, 1, 32.28/cherry_deno, 1]).reshape(-1, 1)
    loss = (1 - np.sum(role, axis=0)).reshape(1, -1)
    p = np.concatenate([role, loss]).T  # no numba

    bet = bet
    out = np.array([3+42+bet, 3+42+bet, 3+42+bet, 3+16+bet, 3+16+bet, 3, 3, 3, 3, 3, 3])
    saf = np.array([294, 294+2, 294+1, 112, 112+2, 14, 10, 8, 2, 3, 0])
    # 適当押し： 重複チェリーの期待枚数は、払い出し x 14/21
    saf = saf + np.array([0, -2+(2*14/21), -1+(14/21), -2+(2*14/21), 0, -2+(2*14/21), 0, 0, 0, 0, 0])

    return p, out, saf

# @njit
def simulate(setting: int, random_state: int = 42)->tuple[float]:
    '''
    入力された設定値と、乱数シード値から遊技機のシュミレーション値を返す
    条件1: 2022年2月実績の平均アウトまで回す
    条件2: ホール割のパラメータ  bet=2.25  cherry_deno=43
    '''
    size = 9000
    out_mean = np.array([7470, 7246, 10350, 11657, 16947, 16659])  # 2022/1実績
    out_d = dict(zip([1, 2, 3, 4, 5, 6], out_mean))

    p, out, saf = imjug(bet=2.25, cherry_deno=43)
    xk = np.arange(len(p.T))
    pk = p[setting-1]
    im = stats.rv_discrete(name='im', values=(xk, pk))
    sample = im.rvs(size=size, random_state=random_state)
    cum = np.cumsum([out[x] for x in sample])
    games = (np.abs(cum - out_d[setting])).argmin()
    result = sample[:games]
    out = cum[games]
    saf = sum([saf[x] for x in result])
    bb = (result < 3).sum()
    rb = ((result > 2) & (result < 5)).sum()

    return tuple(map(float, [bb, rb, games, out, saf]))

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

def  thirty_days(arr: np.ndarray, seed: int = 0) -> pd.DataFrame:
    '''30日 シュミレーション
    '''
    rows = []
    for i, a in enumerate(arr):
        dt = datetime(2022, 4, i+1)
        for j, setting in enumerate(a):
            bb, rb, game, out, saf = simulate(setting, random_state=seed)

def get_result(island, seed=0):
    '''シュミレーション
    '''
    rows = []
    for i, arr in enumerate(island):
        dt = datetime(2023, 11, i+1)
        for j, setting in enumerate(arr):
            bb, rb, game, out, saf = simulate222(setting, random_state=seed)
            row = pd.Series([j+1, bb, rb, game, out, saf], index=('no', 'bb', 'rb', 'game', 'out', 'saf'), name=dt)
            rows.append(row)
            seed += 1

    return pd.DataFrame(rows)

def one_month(num: int, seed: int = 0) -> None:
    pk_even = even_setting_hall()
    arr = distribute_settings_even(pk_even, num, seed=seed)
    df = thirty_days(arr, seed=seed)
    return df

def twenty_four_months(num: int):
    '''
    '''
    rates = []
    for i in range(24):
        df = one_month(num, seed=i*1000)
        rate = df['saf'].sum() / df['out'].sum()
        print(rate)
        rates.append(rate)
    
    return rates


if __name__ == '__main__':

    with timer():
        # df = one_month(16, seed=42)
        # print(df.head())
        # print('rate:', df['saf'].sum() / df['out'].sum())
        rates = twenty_four_months(16)
        print(np.mean(rates))


