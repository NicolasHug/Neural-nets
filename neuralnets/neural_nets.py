import numpy as np
from sklearn.base import BaseEstimator

from . import utils

class NeuralNet(BaseEstimator):
    """A basic neural network class.

    Args:
        n_neurons (list):list of H + 2 integers indicating the number of
            neurons in each layer, including input and output layers. First
            value is the number of features of a training example. Last value
            is either 1 (logistic loss) or C > 1 (cross entropy), where C is
            the number of classes. In-between, the H values indicate the number
            of neurons of each of the H hidden layer.
        activations (str or list of str): The activation functions to use for
            each of the H hidden layers. Allowed values are 'sigmoid', 'tanh',
            'relu' or 'linear' (i.e. no activation). If a str is given, then
            all activations are the same for each hidden layer. If a list of
            string is given, it must be of size H.
        learning_rate(float): The learning rate for gradient descent.
        n_epochs(int): The number of iteration of the gradient descent
            procedure.
        seed(int): A random seed to use for the RNG.
    """

    def __init__(self, n_neurons, activations='relu', learning_rate=.005,
                 n_epochs=10000, seed=None):


        self.lr = learning_rate
        self.n_epochs = n_epochs
        self.losses = []
        self.n_neurons = n_neurons
        self.n_layers = len(self.n_neurons)  # including input and output
        self.m = n_neurons[0]  # dimension of input layer

        if n_neurons[-1] != 1:
            exit('cross entropy is not supported yet')

        self.init_weights(seed)
        self.init_activations(activations)

    def init_activations(self, activations):

        if isinstance(activations, str):  # transform into list
            activations = [activations] * (self.n_layers - 2)
        activations_dict = {
            'sigmoid': (utils.sigmoid, utils.sigmoid_deriv),
            'relu': (utils.relu, utils.relu_deriv),
            'tanh': (utils.tanh, utils.tanh_deriv),
        }
        self.activations = dict()
        self.activations_deriv = dict()
        for i, activation in enumerate(activations):
            try:
                act_fun, deriv_fun = activations_dict[activation]
                self.activations[i + 1] = act_fun
                self.activations_deriv[i + 1] = deriv_fun
            except KeyError:
                exit('Unsupported activation' + activation)

        # Last layer is always a sigmoid activation
        # TODO: change when allowing cross entropy loss
        self.activations[self.n_layers - 1] = utils.sigmoid
        self.activations_deriv[self.n_layers - 1] = utils.sigmoid_deriv

    def init_weights(self, seed):
        if seed is not None:
            np.random.seed(seed)
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
