import cv2
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn import linear_model, svm

def show(img):
    cv2.namedWindow('test', cv2.WINDOW_NORMAL)
    cv2.imshow('test', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def view():
    with open('cl/data.pickle', 'rb') as f:
        data = pickle.load(f)
        print(data[0][0].shape, len(data[0]))  # (784,) 839
        X, y = data
        chars = [a.reshape((28, 28)) for a in X]
        img = np.vstack((
            np.hstack(chars[:40]),
            np.hstack(chars[40:80])
        ))
        print(y[:40])
        print(y[40:80])
        show(img)

# view()

def usemnist():
    import scipy.io as scio

    data = scio.loadmat('mnist.mat')
    # print(data.keys())
    # X = data['trainX'][:3000]  # X は 60000x400 行列
    # y = data['trainY'].ravel()[:3000]  # y は 60000 x 1 行列、ravel()を使って60000次元ベクトルに変換

    x = data['testX']
    y = data['testY'].ravel()
    clf = svm.SVC(gamma=0.001)
    clf.fit(x, y)
    print(clf.score(x, y))

usemnist()

# with open('cl/data.pickle', 'rb') as f:
#     data = pickle.load(f)
#     x, y = data
#     y = [i if not np.isnan(i) else -1 for i in y]

#     x_train, x_test = train_test_split(x)
#     y_train, y_test = train_test_split(y)

#     # clf = MLPClassifier(max_iter=10000)
#     # clf = linear_model.LogisticRegression(penalty='l2', C=10.0) # モデルの定義
#     clf = svm.SVC(gamma=0.001)
#     clf.fit(x_train, y_train)
#     print(clf.score(x_train, y_train))