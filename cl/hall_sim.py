import calendar
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from cl.hall_simulate import simulate
from typing import Any

from contextlib import contextmanager
import time
@contextmanager
def timer():
    t = time.time()
    yield
    print('Elapsed:', time.time() - t)

def filling_vertically(arr: np.ndarray, n: int) -> np.ndarray:
    '''zeros配列を縦に埋めていく、whileは危険か?
    '''
    i = 0
    while n:
        vertical = arr[:, i]
        zero_indices = np.where(vertical==0)[0]
        indices_size = zero_indices.size
        if indices_size > n:
            choice_indices = np.random.choice(zero_indices, size=n, replace=False)
            arr[choice_indices, i] = 1
            n = 0
        else:
            arr[zero_indices, i] = 1
            n = n - indices_size
        i += 1
    return arr

def odd_even():
    dist = np.array([0.417, 0.243, 0.251, 0.046, 0.032, 0.011])  # 2022/1

    pk_odd = dist * np.array([1, 0, 1, 0, 1, 0])  # 奇数設定
    pk_even = dist * np.array([0, 1, 0, 1, 0, 1])  # 偶数設定

    pk_odd = pk_odd + np.array([-0.003, 0, 0, 0, 0, +0.01])  # 調整
    pk_even = pk_even + np.array([0.003, 0, 0, 0, 0, -0.01])

    pk_odd = pk_odd / pk_odd.sum()  # 正規化
    pk_even = pk_even / pk_even.sum()

    return pk_odd, pk_even  # 

def odd_(pk: np.ndarray, num: int, dt: Any, seed=42):
    
    _, days = calendar.monthrange(dt.year, dt.month)
    total = num * days
    dst = list(map(round, total * pk))

    max_idx = dst.index(max(dst))
    diff = total - sum(dst)
    dst[max_idx] = dst[max_idx] + diff  # 余りの台数を一番台数の多い設定で調整する
    print(dst)
    arr = np.zeros((days, num), dtype=np.int64)
    # 設定3
    arr = filling_vertically(arr, dst[2])
    arr[arr==1] = 3
    # 設定5
    arr = filling_vertically(arr, dst[4])
    arr[arr==1] = 5
    # 設定6: 特日
    quotient, remainder = divmod(dst[5], 3)
    for i in [0, 10, 20]:
        horizontal = arr[i, :]
        zero_indices = np.where(horizontal==0)[0]
        horizontal[zero_indices[:quotient]] = 1
    arr = filling_vertically(arr, remainder)
    arr[arr==1] = 6
    # 設定1
    arr[arr==0] = 1
    
    def shuffle_row(row):
        np.random.shuffle(row)
        return row

    shuffled_arr = np.apply_along_axis(shuffle_row, axis=1, arr=arr)
    print(shuffled_arr)


def distribute_settings_(pk: np.ndarray, num: int, dt: datetime, seed=42):
    '''
        偶数設定ホールの設定配分シュミレート
        島の１か月分の設定を返す
    '''
    days = calendar.monthrange(dt.year, dt.month)[1]
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

    rng1 = np.random.RandomState(seed=seed)
    rng1.shuffle(s1_list)
    rng1.shuffle(s2_list)

    for i in [0, 10, 20]:  # 特日の設定2を設定6の分だけマイナス
        s2_list[i] -= s6_list[i]
        for j in range(s6_list[i]):  # マイナス分を分配
            s2_list[i+j+1] += 1

    for i, (s1, s2, s6) in enumerate(zip(s1_list, s2_list, s6_list)):
        if num - (s1 + s2 + s6) >= 0:
            x = [1] * s1 + [2] * s2 + [6] * s6 + [4] * (num - (s1 + s2 + s6))
            rng2 = np.random.RandomState(seed=seed+i)
            rng2.shuffle(x)
            a = np.array(x)
            dst[i] = a
        else:
            print('foo')
    
    return dst

def next_month(dt):
    days = calendar.monthrange(dt.year, dt.month)[1]
    return datetime(dt.year, dt.month, 1) + timedelta(days=days)

def one_month(dst: np.ndarray, dt, seed: int = 0) -> pd.DataFrame:
    '''１か月シュミレーション
    '''
    dt = datetime(dt.year, dt.month, 1)
    days = calendar.monthrange(dt.year, dt.month)[1]
    rows = []
    for i, date in enumerate(dst):
        # print(dt)
        for j, setting in enumerate(date):
            a = simulate(setting, seed=seed)
            row = pd.Series([j+1, *a.tolist()], index=('no', 'bb', 'rb', 'game', 'out', 'saf'), name=dt)
            rows.append(row)
            seed += 1
        dt = dt + timedelta(days=1)

    return pd.DataFrame(rows)

def core(num):
    pk_odd, pk_even = odd_even()
    dst = distribute_settings_even(pk_even, num, seed=42)
    df = one_month(dst, seed=42)
    return df

def main(num: int, months: int, step=1000):
    dt = datetime(2022, 1, 1)
    pk_odd, pk_even = odd_even()
    df = pd.DataFrame()
    for i in range(months):
        dst = distribute_settings_(pk_even, num, dt, seed=i*step)
        df_ = one_month(dst, dt, seed=i*step)
        df = pd.concat([df, df_])
        dt = next_month(dt)
    return df


if __name__ == '__main__':

    with timer():
        # df = core(num=16, step=42)
        # df = main(num=32, months=3, step=1000)
        # print(df)
        # print(df['saf'].sum() / df['out'].sum())

        pk_odd, pk_even = odd_even()
        dt = datetime(2022, 1, 1)
        dst = odd_(pk_odd, 16, dt)

        # n = 23
        # a = np.zeros((7, 7))
        # a[:, 0] = [2,2,2,2,2,2,2]

        # def filling_vertically(arr: np.ndarray, n: int) -> np.ndarray:
        #     '''zeros配列を縦に埋めていく 
        #     '''
        #     i = 0
        #     while n:
        #         vertical = arr[:, i]
        #         zero_indices = np.where(vertical==0)[0]
        #         indices_size = zero_indices.size
        #         if indices_size > n:
        #             choice_indices = np.random.choice(zero_indices, size=n, replace=False)
        #             arr[choice_indices, i] = 1
        #             n = 0
        #         else:
        #             arr[zero_indices, i] = 1
        #             n = n - indices_size
        #         i += 1
        #     return arr
        

        # res = foo(a, n)
        # print(res)

