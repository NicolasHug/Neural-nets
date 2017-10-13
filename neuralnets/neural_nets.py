import numpy as np
from sklearn.base import BaseEstimator

from .utils import sigmoid
from .utils import sigmoid_deriv
from .utils import relu
from .utils import relu_deriv
from .utils import tanh
from .utils import tanh_deriv

class NeuralNet(BaseEstimator):

    def __init__(self, n_neurons, activation='relu', learning_rate=.005,
                 n_epochs=10000, seed=None):

        """
        n_neurons = list of H + 2 integers indicating. First element is the
        number of features of a training example. Last element is either 1
        (logistic loss) or C > 1 (cross entropy). In-between, the H values
        indicate the number of neurons of each hidden layer.
        """

        self.lr = learning_rate
        self.n_epochs = n_epochs
        self.losses = []
        self.n_neurons = n_neurons
        self.n_layers = len(self.n_neurons)  # including input and output
        self.m = n_neurons[0]  # dimension of input layer
        self.seed = seed

        if n_neurons[-1] != 1:
            exit('cross entropy is not supported yet')

        self.activations = dict()
        self.activations_deriv = dict()
        for l in range(1, self.n_layers - 1):
            if activation == 'relu':
                self.activations[l] = relu
                self.activations_deriv[l] = relu_deriv
            elif activation == 'sigmoid':
                self.activations[l] = sigmoid
                self.activations_deriv[l] = sigmoid_deriv
            elif activation == 'tanh':
                self.activations[l] = tanh
                self.activations_deriv[l] = tanh_deriv
            else:
                exit('Unkown activation')

        self.activations[self.n_layers - 1] = sigmoid
        self.activations_deriv[self.n_layers - 1] = sigmoid_deriv

        self.init_weights()

    def init_weights(self):
        if self.seed is not None:
            np.random.seed(self.seed)
        self.W = dict()
        self.b = dict()
        for l in range(1, self.n_layers):
            self.W[l] = np.random.randn(self.n_neurons[l],
                                        self.n_neurons[l - 1]) * .01
            self.b[l] = np.zeros((self.n_neurons[l], 1))

    def fit(self, X, y):
        X = X.T
        self.n = X.shape[1]

        for _ in range(self.n_epochs):

            y_hat, cache = self.forward(X)

            loss = - 1 / self.n  * np.sum(y * np.log(y_hat) +
                                          (1 - y) * np.log(1 - y_hat))
            self.losses.append(loss)

            dW, db = self.backward(X, y, cache)

            for l in range(1, self.n_layers):
                self.W[l] -= self.lr * dW[l]
                self.b[l] -= self.lr * db[l]

    def forward(self, X):

        A = dict()
        Z = dict()

        A[0] = X
        for l in range(1, self.n_layers):
            Z[l] = self.W[l].dot(A[l - 1]) + self.b[l]
            A[l] = self.activations[l](Z[l])

        cache = A, Z  # weights are attributes so no need to cache them

        return A[self.n_layers - 1], cache

    def backward(self, X, y, cache):
        A, Z = cache

        dZ = dict()  # Note: no need to keep dA[l]
        dW = dict()
        db = dict()

        # Backprop last layer
        l = self.n_layers - 1
        dZ[l] = A[l] - y
        dW[l] = 1 / self.n * dZ[l].dot(A[l - 1].T)
        db[l] = 1 / self.n * np.sum(dZ[l], axis=1, keepdims=True)

        # Backprop remaining layers
        for l in reversed(range(1, self.n_layers - 1)):
            dAl = self.W[l + 1].T.dot(dZ[l + 1])
            dAdZ = self.activations_deriv[l](Z[l])
            dZ[l] = dAdZ * dAl
            dW[l] = 1 / self.n * dZ[l].dot(A[l - 1].T)
            db[l] = 1 / self.n * np.sum(dZ[l], axis=1, keepdims=True)

        return dW, db

    def predict(self, X):
        y_hat, _ = self.forward(X.T)
        return (y_hat > .5).squeeze().astype('int')
