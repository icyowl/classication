import numpy as np
import matplotlib.pyplot as plt


X = np.array([[0,0],[0,1],[1,0],[1,1]])
Y = np.array([[0],[0],[0],[1]])


class SimplePerceptron:

    def __init__(self, isRegression = False, times = 1000, alpha = 0.5, epsilon = 0.0001, seed=0):
        """
        初期化
        
        Args:
            isRegression (bool) : 回帰の場合はTrue
            times        (int)  : 学習回数
            alpha        (float): 学習率
            epsilon      (float): 学習を終了する閾値
            seed         (int)  : 乱数の種
        """
        self.isRegression = isRegression
        self.W = None
        self.b = None
        self.times = times
        self.alpha = alpha
        self.epsilon = epsilon
        np.random.seed(seed)

    def Forward(self, X):
        # 活性化関数への入力を計算しZに代入
        Z = X.dot(self.W) + self.b

        a = self.ActivationFunction(Z)

        if not self.isRegression:
            # 0.5以上の項は1.0に、0.5未満の項は0.0にする
            return np.where(a > 0.5, 1.0, 0.0)

        else:
            return a

    def Sigmoid(self, value):
        return 1.0/(1.0 + np.exp(- value))

    def Sigmoid_d(self, value):
        """
        シグモイド関数(ロジスティック関数)の微分.
        """
        return self.Sigmoid(value) * (1.0 - self.Sigmoid(value))

    def ActivationFunction(self, value):
        return self.Sigmoid(value)

    def ActivationFunction_d(self, value):
        """
        活性化関数の微分.
        今回は活性化関数としてシグモイド関数を採用.
        """
        return self.Sigmoid_d(value)

    def J(self, Yh, Y):
        # コスト関数を計算しjに代入
        j = ((Yh - Y) * (Yh - Y)).sum() / 2.0

        return j

    def Delta(self, X, Y):
        m, n = X.shape

        # dw(n×1の行列)を0で初期化
        dw = np.zeros((n,1))
        db = np.zeros((1,1))

        Z = X.dot(self.W) + self.b
        Yh = self.ActivationFunction(Z)
        G = self.ActivationFunction_d(Z)
        for i in range(m):

            # バイアス項の偏微分を計算
            db[0][0] = db[0][0] + (Yh[i]  - Y[i]) * G[i]

            for j in range(n):
                # 重みを計算しdw[j][0]に足し合わせる
                dw[j][0] = dw[j][0] + (Yh[i] - Y[i]) * G[i] * X[i][j]

        return dw, db

    def Fit(self, X, Y):

        # 重みとバイアス項の初期化
        self.W = np.random.rand(X.shape[1], 1)
        self.b = np.zeros((1,1))

        # コスト関数の値がどう変わったのかを保存するリストを初期化
        J_List = []

        # 重み更新ループ
        for t in range(self.times):
            Yh = self.ActivationFunction(X.dot(self.W) + self.b)

            # 誤差の計算
            j = self.J(Yh, Y)
            J_List.append(j)

            # 誤差がepsilon以下なら終了
            if j <= self.epsilon:
                break

            # 重みとバイアス項の更新
            dw, db = self.Delta(X, Y)
            self.W = self.W - self.alpha * dw
            self.b = self.b - self.alpha * db            

        return J_List

if __name__ == '__main__':

    SP = SimplePerceptron()
    l = SP.Fit(X, Y)

    # コスト関数が小さくなっていることの確認
    plt.plot(l)
    plt.xlabel("times")
    plt.ylabel("error")
    plt.show()

