import numpy as np
np.set_printoptions(suppress=True, precision=7)


def imjug(bet=2.25, cherry_deno=48.42):
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


if __name__ == '__main__':
    from scipy import stats

    p, out, saf = imjug(bet=2.25, cherry_deno=43)
    
    setting=1
    seed=41
    size=8000000
    xk = np.arange(p.shape[1])
    pk = p[setting-1]
    im = stats.rv_discrete(name='im', values=(xk, pk))  # no numba
    sample = im.rvs(size=size, random_state=seed)
    out_ = [out[x] for x in sample]
    saf_ = [saf[x] for x in sample]

    print(sum(saf_) / sum(out_))