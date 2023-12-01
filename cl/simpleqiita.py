#内積
def dot(vec0,vec1):
    tmp = 0
    for i, j in zip(vec0,vec1):
        tmp += i * j
    return tmp
#step関数
def step(num):
    if num > 0:
        return 1
    else:
        return 0
#出力
def feedforward(i,w):
    return step(dot(i,w))
#逐次学習
def train(w,i,y,eta):
    o = feedforward(i,w)
    for j in range(len(w)):
        w[j] = w[j] + (y - o) * i[j] * eta
    return w

#main処理,andを学習させる。
if __name__ == "__main__":
    train_x = [[0,0,1],[0,1,1],[1,0,1],[1,1,1]]
    train_y = [0,0,0,1]
    weight  = [0,0,0]
    eta     = 0.1

    epoch = 100
    for i in range(epoch):
        for x,y in zip(train_x,train_y):
            weight = train(weight,x,y,eta)

    #確認,0,0,0,1が出力されれば大丈夫
    for x in train_x:
        print(feedforward(x,weight))