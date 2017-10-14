import numpy as np
from sklearn.datasets import make_blobs
from sklearn.datasets import make_moons

from neuralnets import NeuralNet


def test_check_gradients():
    '''Gradient-check for NN on various datasets, architecture and loss
    functions, with batch learning and regularization.
    '''

    learning_rate = .005
    n_epochs = 1000
    batch_size = 64
    lambda_reg = .1

    seed = 0

    X, y = make_blobs(n_samples=100, centers=3, n_features=2,
                      random_state=seed)
    nn = NeuralNet(n_neurons=[2, 2, 3],
                   activations='tanh',
                   learning_rate=learning_rate,
                   n_epochs=n_epochs,
                   batch_size=64,
                   lambda_reg=lambda_reg,
                   check_gradients=True,
                   seed=seed)
    nn.fit(X, y)  # would raise exception if grad check fails

    X, y = make_moons(n_samples=100, noise=.1, random_state=seed)
    nn = NeuralNet(n_neurons=[2, 2, 2, 1],
                   activations='tanh',
                   learning_rate=learning_rate,
                   n_epochs=n_epochs,
                   batch_size=64,
                   lambda_reg=lambda_reg,
                   check_gradients=True,
                   seed=seed)
    nn.fit(X, y)  # would raise exception if grad check fails
