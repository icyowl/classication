import calendar
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from cl.hall_simulate import simulate

from contextlib import contextmanager
import time
@contextmanager
def timer():
    t = time.time()
    yield
    print('Elapsed:', time.time() - t)

def next_month(dt):
    days = calendar.monthrange(dt.year, dt.month)[1]
    return datetime(dt.year, dt.month, 1) + timedelta(days=days)

def filling_vertically(arr: np.ndarray, n: int) -> np.ndarray:
    '''縦に埋めていく、
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

def odd_distribute(pk: np.ndarray, num: int, dt: datetime, seed=42):
    
    days = calendar.monthrange(dt.year, dt.month)[1]
    total = num * days
    dst = list(map(round, total * pk))

    idx_max = dst.index(max(dst))
    diff = total - sum(dst)
    print(diff)
    dst[idx_max] = dst[idx_max] + diff  # 余りの台数を一番台数の多い設定で調整する

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
    
    rng = np.random.RandomState(seed=seed)

    def shuffle_row(row):
        rng.shuffle(row)
        return row

    return np.apply_along_axis(shuffle_row, axis=1, arr=arr)

def even_distribute(pk: np.ndarray, num: int, dt: datetime, seed=42):
    days = calendar.monthrange(dt.year, dt.month)[1]
    total = num * days
    dst = list(map(round, total * pk))

    idx_max = dst.index(max(dst))
    diff = total - sum(dst)
    dst[idx_max] = dst[idx_max] + diff  # 余りの台数を一番台数の多い設定で調整する

    arr = np.zeros((days, num), dtype=np.int64)
    # 設定6: 特日
    quotient, remainder = divmod(dst[5], 3)
    for i in [0, 10, 20]:
        horizontal = arr[i, :]
        zero_indices = np.where(horizontal==0)[0]
        horizontal[zero_indices[:quotient]] = 1
    arr[arr==1] = 6
    # 設定2
    arr = filling_vertically(arr, dst[1])
    arr[arr==1] = 2
    # 設定4
    arr = filling_vertically(arr, dst[3])
    arr[arr==1] = 4
    # 設定1
    arr[arr==0] = 1

    rng = np.random.RandomState(seed=seed)

    def shuffle_row(row):
        rng.shuffle(row)
        return row

    return np.apply_along_axis(shuffle_row, axis=1, arr=arr)


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
    dt = datetime(2022, 1, 1)
    dst = odd_distribute(pk_odd, num, dt, seed=42)
    df = one_month(dst, dt, seed=42)
    return df

def main(num: int, months: int, step=1000):
    dt = datetime(2022, 1, 1)
    pk_odd, pk_even = odd_even()
    df = pd.DataFrame()
    for i in range(months):
        # dst = odd_distribute(pk_odd, num, dt, seed=i*step)
        dst = even_distribute(pk_even, num, dt, seed=i*step)
        df_ = one_month(dst, dt, seed=i*step)
        df = pd.concat([df, df_])
        dt = next_month(dt)
    return df


if __name__ == '__main__':

    with timer():
        df = core(num=16)
        # df = main(num=16, months=24, step=1000)
        print(df)
        print(df['saf'].sum() / df['out'].sum())

        # pk_odd, pk_even = odd_even()
        # dt = datetime(2022, 1, 1)
        # dst = odd_distribute(pk_odd, num=16, dt=dt, seed=42)
        # # print(dst)
        # dst = even_distribute(pk_even, num=16, dt=dt, seed=42)
        # print(dst)



