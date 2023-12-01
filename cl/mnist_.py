import matplotlib.pyplot as plt
import scipy.io as scio
from sklearn import linear_model

# scipy.io.loadmat()を使ってmatlabデータを読み込み
data = scio.loadmat('mnist.mat')
# print(data.keys())
# X = data['trainX'][:3000]  # X は 60000x400 行列
# y = data['trainY'].ravel()[:3000]  # y は 60000 x 1 行列、ravel()を使って60000次元ベクトルに変換

_X = data['testX']
_y = data['testY'].ravel()

a = _X[1]
print(a.reshape(28, 28))
# plt.axis('off')
# plt.imshow(a.reshape(28, 28), cmap='gray')
# plt.show()