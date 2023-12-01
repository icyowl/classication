# 4.1.3 分類における最小二乗（p.182）
import numpy as np
import matplotlib.pyplot as plt
# from pylab import *
import sys

K = 2    # 2クラス分類
N = 100  # データ数

def f(x1, W_t):
    # 決定境界の直線の方程式
    a = - ((W_t[0,1]-W_t[1,1]) / (W_t[0,2]-W_t[1,2]))
    b = - (W_t[0,0]-W_t[1,0])/(W_t[0,2]-W_t[1,2])
    return a * x1 + b

if __name__ == "__main__":
    # 訓練データを作成
    cls1 = []
    cls2 = []
    
    # データは正規分布に従って生成
    mean1 = [-1, 2]  # クラス1の平均
    mean2 = [1, -1]  # クラス2の平均
    cov = [[1.0,0.8], [0.8,1.0]]  # 共分散行列（全クラス共通）

    # ノイズなしデータ
    cls1.extend(np.random.multivariate_normal(mean1, cov, int(N/2)))
    cls2.extend(np.random.multivariate_normal(mean2, cov, int(N/2)))

    # データ行列Xを作成
    temp = np.vstack((cls1, cls2))
    temp2 = np.ones((N, 1))  # バイアスw_0用に1を追加
    X = np.hstack((temp2, temp))
    
    # ラベル行列T（1-of-K表記）を作成
    T = []
    for i in range(int(N/2)):
        T.append(np.array([1, 0]))  # クラス1
    for i in range(int(N/2)):
        T.append(np.array([0, 1]))  # クラス2
    T = np.array(T)
    
    # パラメータ行列Wを最小二乗法で計算（式4.16）
    X_t = np.transpose(X)
    temp = np.linalg.inv(np.dot(X_t, X))  # 行列の積はnp.dot(A, B)
    W = np.dot(np.dot(temp, X_t), T)
    W_t = np.transpose(W)
    print(W_t)
    
    # 訓練データを描画
    x1, x2 = np.transpose(np.array(cls1))
    plt.plot(x1, x2, 'rx')
    
    x1, x2 = np.transpose(np.array(cls2))
    plt.plot(x1, x2, 'bo')
    
    # 識別境界を描画
    x1 = np.linspace(-4, 8, 1000)
    x2 = [f(x, W_t) for x in x1]
    plt.plot(x1, x2, 'g-')
    
    plt.xlim(-4, 8)
    plt.ylim(-8, 4)
    plt.show()