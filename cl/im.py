from scipy import stats
import numpy as np


def imjug() -> tuple[np.ndarray]:
    """アイムジャグラーの確率表と配当表を配列化する
    Returns:
        tuple[np.ndarray]:
            pk.shape  -> (6, 8) 確率表
            out.shape -> (8, 1) IN枚数
            saf.shape -> (8, 1) 払い出し
    """
    big = 273.1, 269.7, 269.7, 259.0, 259.0, 255.0
    reg = 439.8, 399.6, 331.0, 315.1, 255.0, 255.0
    bell =  (1092.27,) * 6
    clown = (1092.27,) * 6
    grape = 6.02, 6.02, 6.02, 6.02, 6.02, 5.78
    cherry =  (32.28,) * 6
    replay = (7.298,) * 6

    p = 1/np.array([big, reg, bell, clown, grape, cherry, replay])
    q = 1 - np.sum(p, axis=0).reshape(1, -1)  # ハズレ
    pk = np.concatenate([p, q]).T

    bet = 0.75  # ボーナス揃えのBET、25%の先告知は直揃え
    out = np.array([3+42+bet, 3+16+bet, 3., 3., 3., 3., 3., 3.])
    saf = np.array([294., 112., 14., 10., 8., 2., 3., 0.])

    return pk, out, saf




# p, out, saf = imjug()

# rates = (p * saf).sum(axis=1) / (p * out).sum(axis=1)
# print(rates.reshape(-1, 1) * 100)

# if __name__ == '__main__':

from scipy import stats
import matplotlib.pyplot as plt

setting = 6
pk, out, saf = imjug()

xk = np.arange(pk.shape[1])
pk_ = pk[setting-1]
im = stats.rv_discrete(name='im', values=(xk, pk_))

plt.style.use('seaborn-v0_8-darkgrid')
fig, ax = plt.subplots()
ax.plot(xk, im.pmf(xk), 'co', ms=12)
ax.vlines(xk, 0, im.pmf(xk), colors='c', lw=4)

ax.set_title(f'Im pmf (Setting {setting})')
ax.set_yscale('log')
labels = 'big', 'reg', 'grape', 'cherry', 'bell', 'clown', 'replay', 'lose'
ax.set_xticks(xk, labels, rotation=90)
# plt.show()

# size = 8000
# seed = 0
# sample = im.rvs(size=size, random_state=seed)
# result = [(saf - out)[x] for x in sample]

# plt.style.use('seaborn-v0_8-darkgrid')
# fig, ax = plt.subplots()
# x = np.arange(size)
# y = np.cumsum(result)
# ax.plot(x, y)
# ax.set_title(f'setting 6 and random_state={seed}')
# plt.show()

plt.savefig('im6-pmf')

# exp = (pk * saf).sum(axis=1) - (pk * out).sum(axis=1)
# print(exp.reshape(-1, 1) * 8000)