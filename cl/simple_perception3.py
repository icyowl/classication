import numpy as np
import seaborn as sns


class Perceptron:
    def __init__(self, eta=0.1, n_iter=1000):
        self.eta=eta
        self.n_iter=n_iter
        self.w = np.array([])

    @property
    def w_(self):
        return self.w

    def predict(self, x):
        x_ = np.hstack([1., x])
        return 1 if self.w.T @ x_ >= 0 else -1

    def fit(self, x, y):
        self.w = np.ones(len(x[0])+1)
        x = np.hstack([np.ones((len(x),1)), x])
        for _ in range(self.n_iter):
            for i in range(len(x)):
                loss = np.max([0, -y[i] * self.w.T @ x[i]])
                if loss != 0:
                    self.w += self.eta * y[i] * x[i]

if __name__ == '__main__':

    df = sns.load_dataset('iris')
    df = df[df['species']!='versicolor']
    df['species'] = df['species'].map({'setosa':1, 'virginica':-1})

    x = df.iloc[:,0:2].values
    y = df['species'].values

    model = Perceptron()
    model.fit(x, y)
    # print(model.w_)
    for a in x:
        print(model.predict(a))
