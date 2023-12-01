import numpy as np
import matplotlib.pyplot as plt
import scipy.io as scio
from sklearn import linear_model

# scipy.io.loadmat()を使ってmatlabデータを読み込み
data = scio.loadmat('mnist.mat')
# print(data.keys())
X = data['trainX'][:3000]  # X は 5000x400 行列
y = data['trainY'].ravel()[:3000]  # y は 5000 x 1 行列、ravel()を使って5000次元ベクトルに変換
# print(len(X))
model = linear_model.LogisticRegression(penalty='l2', C=10.0, max_iter=1000) # モデルの定義
model.fit(X, y)    # 訓練データで学習

_X = data['testX']
_y = data['testY'].ravel()
res = model.predict(_X)
print(res)
print(_y)
print(model.score(_X, _y))