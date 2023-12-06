import numpy as np

@np.vectorize
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def verticalize(row):
    return np.reshape(row, (1, len(row)))

rho = 1

x = np.array([[0, 0, -1], [0, 1, -1], [1, 0, -1], [1, 1, -1]])
y = np.array([[1, 0], [0, 1], [0, 1], [1, 0]])

w1 = np.random.randn(3, 2)
w2 = np.random.randn(3, 2)

for i in range(50000):
    for p in range(len(x)):
        xp = verticalize(x[p])
        yp = y[p]
        g1 = sigmoid(xp @ w1)
        g1 = verticalize(np.hstack((g1[0], [-1])))
        g2 = sigmoid(g1 @ w2)
        eps_out = (g2 - yp) * g2 * (1 - g2)
        eps_hidden = np.delete(np.sum(eps_out*w2, axis=1)*g1*(1 - g1), -1, 1)
        w2 -= rho * g1.T @ eps_out
        w1 -= rho * xp.T @ eps_hidden

for p in range(len(x)):
    xp = verticalize(x[p])
    yp = y[p]
    g1 = sigmoid(xp @ w1)
    g1 = verticalize(np.hstack((g1[0], [-1])))
    g2 = sigmoid(g1 @ w2)
    print(g2[0])
    print(np.argmax(g2))