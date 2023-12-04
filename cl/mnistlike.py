import cv2
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn import linear_model, svm

def show(img):
    # cv2.namedWindow('test', cv2.WINDOW_NORMAL)
    cv2.imshow('test', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def view():
    with open('cl/data.pickle', 'rb') as f:
        data = pickle.load(f)
        print(len(data[0]))  # (784,) 839
        width = 30
        for i in range(10):
            X = [x for x, y in zip(*data) if y == i]
            chars = [a.reshape((28, 28)) for a in X]
            remain = len(chars) % width
            chars = chars + [np.zeros((28, 28))] * (width - remain)
            hstacks = [np.hstack(chars[i:i+width]) for i in range(0, len(chars), width)]
            img = np.vstack(hstacks)
            show(img)

view()

def usemnist():
    import scipy.io as scio

    data = scio.loadmat('mnist.mat')
    # print(data.keys())
    # X = data['trainX'][:3000]  # X は 60000x400 行列
    # y = data['trainY'].ravel()[:3000]  # y は 60000 x 1 行列、ravel()を使って60000次元ベクトルに変換

    x = data['testX'][:769]
    y = data['testY'].ravel()[:769]

    clf = svm.SVC(gamma=0.001)
    clf.fit(x, y)
    print(clf.score(x, y))

# usemnist()

def fitfit():
    with open('cl/data.pickle', 'rb') as f:
        data = pickle.load(f)

        data_ = [(x, y) for x, y in zip(*data) if not np.isnan(y)]
        X, y = zip(*data_)
        # print(len(x))
        # X, y = data
        # y = [e if not np.isnan(e) else -1 for e in y]

        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)


        clf = MLPClassifier(max_iter=10000)
        clf = linear_model.LogisticRegression(penalty='l2', C=10.0) # モデルの定義
        # clf = svm.SVC(gamma=0.001)
        clf.fit(X_train, y_train)
        print(clf.score(X_train, y_train))
        print(clf.score(X_test, y_test))