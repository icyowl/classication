import numpy as np
import matplotlib.pyplot as plt

# ネットワークモデルに必要な活性化関数の定義(sigmoid関数)
def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_grad(x):
    return (1.0 - sigmoid(x))*sigmoid(x)

# ネットワークモデルの定義(単純パーセプトロン)
class Perceptron:
    def __init__(self, input_size, weight_init_std=0.01):
        # 平均0，分散1の正規分布の乱数を生成
        self.w = weight_init_std * np.random.randn(input_size)
        # バイアスの定義
        self.b = 0.0
        # 傾きを保存しておくリストを作成
        self.grads = {}

    def forward(self, x): # x:入力
        self.a = np.dot(x, self.w) + self.b
        self.y = sigmoid(self.a)
        return self.y

    def backward(self, x, t): # x:入力, y:教師
        # 傾きを保存しておくリストを初期化
        self.grads = {}
        # 入力データと教師ラベルとの誤差を計算し損失を算出
        l = -1 * (t - self.y)
        # sigmoid関数地点での勾配を算出
        g = l * sigmoid_grad(self.y)
        # それぞれのパラメータの箇所での勾配を算出
        self.grads['w'] = np.dot(x, g)
        self.grads['b'] = 1 * g

    def update_parameters(self, lr=0.1): # lr:学習率
        self.w -= lr * self.grads['w']
        self.b -= lr * self.grads['b']

# モデルパラメータを表示させる関数
def display_model_parameters(model):
    print("w : ", model.w, "b : ", model.b)


if __name__=="__main__":
    # AND回路の学習を行うための入力データと教師ラベルの定義
    input_data = np.array([[0,0],[0,1],[1,0],[1,1]])
    label_data = np.array([0,0,0,1])

    # モデルの作成
    input_size = 2
    model = Perceptron(input_size=input_size, weight_init_std=1)
    display_model_parameters(model)

    # 学習パラメータの指定
    num_train_data = 4
    epoch_num = 5000
    learning_rate = 0.1
    train_loss_list = []

    # 学習回数分繰り返す
    for epoch in range(1, epoch_num+1, 1):
        sum_loss = 0.0
        for i in range(0, num_train_data, 1):
            # 学習1回に用いるinputデータとlabelを抽出
            input = input_data[i]
            label = label_data[i]

            y_pred = model.forward(input)
            model.backward(input, label)
            model.update_parameters(lr=learning_rate)

            sum_loss += np.power(y_pred - label ,2)
        
        train_loss_list.append(sum_loss/4)
        print("epoch : {}, loss : {}" .format(epoch, sum_loss/4))

    display_model_parameters(model)

    #正解率の算出
    print("=============検証==============")
    cnt_correct = 0
    cnt_all = 0
    tolerance = 0.1 #許容範囲の設定
    for i in range(0,len(input_data)):
        y = model.forward(input_data[i])
        print("input_data : {}, y : {}".format(input_data[i], y))
        label = label_data[i]
        if  label-tolerance < y and y < label+tolerance:
            cnt_correct += 1
        cnt_all += 1

    accuracy = cnt_correct/cnt_all
    print("accuracy : ", accuracy)

    #学習推移グラフの描画
    plt.plot(range(len(train_loss_list)), train_loss_list)
    plt.xlabel('epoch')
    plt.ylabel('train_loss')
    plt.show()
