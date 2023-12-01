import numpy as np
import matplotlib.pyplot as plt
import scipy.io as scio
from sklearn import linear_model

# scipy.io.loadmat()を使ってmatlabデータを読み込み
data = scio.loadmat('mnist.mat')
X = data['X']  # X は 5000x400 行列
y = data['y'].ravel()  # y は 5000 x 1 行列、ravel()を使って5000次元ベクトルに変換

model = linear_model.LogisticRegression(penalty='l2', C=10.0) # モデルの定義
model.fit(X, y)    # 訓練データで学習
model.score(X, y)  # 訓練データでの正答率
