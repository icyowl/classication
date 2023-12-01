import matplotlib.pyplot as plt
#内積
def dot(input_x, weight_):
    tmp = 0
    for i, j in zip(input_x, weight_):
        tmp += i * j
    return tmp
#step関数
def step(num):
    if num > 0:
        return 1
    else:
        return 0
#出力
def forward(input_x, weight_):
    return step(dot(input_x, weight_))
#逐次学習
def train(weight_, input_x,train_y, eta):
    output_ = forward(input_x, weight_)
    for j in range(len(weight_)):
        weight_[j] = weight_[j] + (train_y - output_) * input_x[j] * eta
    return weight_

#main処理,andを学習させる。
if __name__ == "__main__":
    train_x = [[0,0,1],[0,1,1],[1,0,1],[1,1,1]]
    train_y = [0,0,0,1]
    weight  = [0,0,0]
    eta     = 1. # 学習係数
    # 重みを更新するとき、どのくらい変化させるかを表す数字.
    # 学習係数は1でも一応収束（学習がきちんと終わること）するし、
    # 重みベクトルがすべて0だった時は、学習係数がいくつだろうと
    # 収束に掛かる回数は同じなので、省けます。

    epoch = 10
    y_ = []
    for i in range(epoch):
        for x, y in zip(train_x, train_y):
            weight = train(weight, x, y, eta)
            w = weight.copy()
            y_.append(w)
    #確認,0,0,0,1が出力されれば大丈夫
    print(weight)
    for x in train_x:
        print(forward(x, weight))

    x = range(len(y_))
    plt.plot(x, y_)
    plt.show()