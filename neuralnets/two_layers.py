import numpy as np
from sklearn.base import BaseEstimator

from .utils import sigmoid
from .utils import sigmoid_deriv
from .utils import relu
from .utils import relu_deriv
from .utils import tanh
from .utils import tanh_deriv


class TwoLayersNN(BaseEstimator):

    def __init__(self, n_h=5, activation='relu', learning_rate=.005,
                 n_epochs=1000, seed=None):
        self.lr = learning_rate
        self.n_epochs = n_epochs
        self.n_h = n_h
        self.losses = []
        self.seed = seed

        if activation == 'relu':
            self.activation = relu
            self.activation_deriv = relu_deriv
        elif activation == 'sigmoid':
            self.activation = sigmoid
            self.activation_deriv = sigmoid_deriv
        elif activation == 'tanh':
            self.activation = tanh
            self.activation_deriv = tanh_deriv
        else:
            exit('Unkown activation')

    def fit(self, X, y):
        X = X.T
        dim, self.n = X.shape

        if self.seed is not None:
            np.random.seed(self.seed)
        self.W1 = np.random.randn(self.n_h, dim) * .01
        self.b1 = np.zeros((self.n_h, 1))
        self.W2 = np.random.randn(1, self.n_h) * .01
        self.b2 = np.zeros((1, 1))

        for _ in range(self.n_epochs):


            y_hat, cache = self.forward(X)

            loss = - 1 / self.n  * np.sum(y * np.log(y_hat) +
                                          (1 - y) * np.log(1 - y_hat))
            self.losses.append(loss)

            dW1, db1, dW2, db2 = self.backward(X, y, cache)

            self.W1 -= self.lr * dW1
            self.b1 -= self.lr * db1
            self.W2 -= self.lr * dW2
            self.b2 -= self.lr * db2

    def forward(self, X):
        Z1 = self.W1.dot(X) + self.b1
        A1 = self.activation(Z1)
        Z2 = self.W2.dot(A1) + self.b2
        A2 = sigmoid(Z2)
        cache = (self.W1, self.b1, Z1, A1, self.W2, self.b2, Z2, A2)

        return A2, cache

    def backward(self, X, y, cache):
        """
        Li = -[y(i) log(a(i)) + (1 - y(i)) log(1 - a(i))]
        L = 1 / n sum_{i = 1}^n L_i

        So dL / dw = 1 / n sum_{i = 1}^n dL_i / dw
        """
        W1, b1, Z1, A1, W2, b2, Z2, A2 = cache

        dZ2 = A2 - y  # dLi / dZ2  (each dLi, obviously)
        dW2 = 1 / self.n * dZ2.dot(A1.T)  # dL / dZ2
        db2 = 1 / self.n * np.sum(dZ2, axis=1, keepdims=True)

        dA1 = W2.T.dot(dZ2)
        dA1Z1 = self.activation_deriv(Z1)
        dZ1 = dA1 * dA1Z1
        dW1 = 1 / self.n * dZ1.dot(X.T)
        db1 = 1 / self.n * np.sum(dZ1, axis=1, keepdims=True)

        return dW1, db1, dW2, db2

    def predict(self, X):
        a, _ = self.forward(X.T)
        return (a > .5).squeeze().astype('int')
