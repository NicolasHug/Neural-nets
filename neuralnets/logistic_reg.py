import numpy as np
from sklearn.base import BaseEstimator

from .utils import sigmoid


class LogisticReg(BaseEstimator):
    """Logistic regression model. Mostly useful to check the more general
    NeuralNet class."""

    def __init__(self, learning_rate=.005, n_epochs=10000, seed=None):
        self.lr = learning_rate
        self.n_epochs = n_epochs
        self.losses = []
        self.seed = seed

    def fit(self, X, y):
        X = X.T
        dim, self.n = X.shape

        if self.seed is not None:
            np.random.seed(self.seed)
        self.w = np.random.randn(dim, 1) * .01
        self.b = 0
        for _ in range(self.n_epochs):
            y_hat, (z, a) = self.forward(X)

            loss = - 1 / self.n  * np.sum(y * np.log(a) +
                                          (1 - y) * np.log(1 - a))
            self.losses.append(loss)

            dw, db = self.backward(X, y, z, a)

            self.w -= self.lr * dw
            self.b -= self.lr * db

    def forward(self, X):
        '''Forward pass. Return the output y_hat and the intermediate
        computations. (Here a = y_hat)'''
        z = self.w.T.dot(X) + self.b
        a = sigmoid(z)  # y_hat
        cache = (z, a)

        return a, cache

    def backward(self, X, y, z, a):
        '''Backward propagation. Returns the gradients.
        Li = -[y(i) log(a(i)) + (1 - y(i)) log(1 - a(i))]
        L = 1 / n sum_{i = 1}^n L_i
        '''
        #da = - (y / a - (1 - y) / (1 - a))  # dLi / da(i)
        #dz = sigmoid(z) * (1 - sigmoid(z)) * da  # dLi / dz(i)
        # or more directly:
        # dz = a * (1 - a) * da = a - y
        dz = a - y
        dw = dz * X  # dLi / dw
        dw = 1 / self.n * dw.sum(axis=1, keepdims=True)  # dL / dw
        db = 1 / self.n * dz.sum()  # dL / db

        return dw, db

    def predict(self, X):
        a, _ = self.forward(X.T)
        return (a > .5).squeeze().astype('int')
